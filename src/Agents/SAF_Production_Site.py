from src.Agents.Feedstock_Aggregator import FeedstockAggregator

from mesa import Agent

from typing import List, Any, Optional, TYPE_CHECKING

import logging

import random

# Contract type import (avoids circular dependency)
if TYPE_CHECKING:
    from src.Agents.FeedstockContract import FeedstockContract

 

logger = logging.getLogger(__name__)

 

class SAFProductionSite(Agent):

    """

    SAF production site agent.

 

    Responsibilities:

      - Represents a physical plant under construction or operation.

      - Derives SRMC from feedstock, transport, opex, and profit margin inputs.

      - Computes potential production output based on design and realised load factors.

            - Design: Decided at investment based on available feedstock to be pledged.

            - Realised: Adjusted based on variable feedstock availability.

 

    Attributes:

        site_id (str): Unique identifier for the site.

        state_id (str): State in which the site operates.

        investor_id (str): Owning investor identifier.

        max_capacity (float): Maximum possible production capacity for a plant.

        design_load_factor (float): Fixed design load factor chosen at investment (0 < dlf ≤ 1) based on available feedstock in state.

        opex (float): Operating cost component.

        aggregator (FeedstockAggregator): Aggregator of state where SAF Production Plant sits.

        capex_schedule (List[float]): CAPEX outlay schedule.

        transport_cost (float): Per-unit feedstock transport cost.

        profit_margin (float): Per-unit margin added (used in SRMC for market merit ordering).

        construction_years (int): Planned construction duration.

        tick_built (int): Simulation tick construction started.

        operational_year (int): Tick when plant becomes operational (post CAPEX schedule).

        srmc (float): Short-run marginal cost (includes profit margin).

        tick_production_output (float): Production output this current tick.

    """

 

    site_counter: int = 0

 

    def __init__(

        self,

        unique_id: Any,

        model: Any,

        state_id: str,

        investor_id: str,

        config: dict,

        design_load_factor: float,

        aggregator: FeedstockAggregator,

        capex_schedule: list,

    ) -> None:

        super().__init__(unique_id, model)

        """

        Initialise a SAFProductionSite.

 

        Args:

            unique_id: Mesa agent unique identifier.

            model: SAF Market Model.

            state_id: State identifier.

            investor_id: Owning investor identifier.

            config: Global configuration dictionary.

            design_load_factor: Fixed design load factor (0 < value ≤ 1).

            aggregator: FeedstockAggregator supplying feedstock conditions.

            capex_schedule: CAPEX list; empty implies already operational.

        """

 

        super().__init__(unique_id, model)

 

        self.site_id: str = self.unique_id

        self.state_id: str = state_id

        self.investor_id: str = investor_id

        self.max_capacity: float = config["max_capacity"]

        self.design_load_factor: float = design_load_factor

        self.opex: float = config["opex"]

        self.aggregator: FeedstockAggregator = aggregator

        self.capex_schedule: List = capex_schedule

        self.transport_cost: float = config["transport_cost"]

        self.profit_margin: float = config["profit_margin"]

        self.construction_years: int = model.config["saf_plant_construction_time"]

        self.tick_built: int = model.schedule.time

        self.operational_year = round(self.tick_built + len(self.capex_schedule))

        self.srmc = self.calculate_srmc()

        self.streamday_percentage = self.sample_streamday_percentage()

        # Link to feedstock contract (enables priority allocation)
        self.active_contract: Optional['FeedstockContract'] = None



    def sample_streamday_percentage(self) -> float:

        """

        Sample the fraction of days the plant is operating (on average).

 

        Returns:

            Random multiplier in [streamday_min, streamday_max].

        """

        streamday_min = self.model.config["streamday_min"]

        streamday_max = self.model.config["streamday_max"]

        multiplier = random.uniform(streamday_min, streamday_max)

        return multiplier

 

    def calculate_srmc(self, current_year: int = None) -> float:
        """
        Compute short-run marginal cost (SRMC): Cost to produce 1 tonne of SAF.

        Formula: SRMC = feedstock_price + opex + transport_cost + profit_margin

        If current_year provided, all cost components escalate at market_escalation_rate.
        Otherwise uses base prices (backwards compatibility).

        Parameters:
            current_year: Optional calendar year for market price escalation

        Returns:
            SRMC value (float).
        """
        if current_year is not None:
            start_year = int(self.model.config["start_year"])
            years_elapsed = current_year - start_year
            market_escalation_rate = float(self.model.config.get("market_escalation_rate", 0.02))
            escalation_factor = (1 + market_escalation_rate) ** years_elapsed

            feedstock_price = self.aggregator.get_current_market_price(current_year)
            opex_escalated = self.opex * escalation_factor
            transport_escalated = self.transport_cost * escalation_factor
            margin_escalated = self.profit_margin * escalation_factor

            srmc_total = feedstock_price + opex_escalated + transport_escalated + margin_escalated
        else:
            feedstock_price = self.aggregator.feedstock_price
            srmc_total = feedstock_price + self.opex + self.transport_cost + self.profit_margin

        return srmc_total



    def get_contracted_capacity(self, current_year: int = None) -> float:
        """
        Calculate contracted capacity (contract_percentage × max_capacity).
        Returns 0.0 if no active contract.
        """
        if self.active_contract is None:
            return 0.0

        if current_year is not None and not self.active_contract.is_active(current_year):
            return 0.0

        return self.max_capacity * self.active_contract.contract_percentage

    def get_spot_capacity(self, current_year: int = None) -> float:
        """Calculate spot (non-contracted) capacity."""
        return self.max_capacity - self.get_contracted_capacity(current_year)

    def get_spot_utilization_factor(self) -> float:
        """
        Decide how much of spot capacity to utilize based on price signals.

        SPOT CAPACITY OPTIMIZATION:
        Plant compares spot price vs contract price and decides whether to:
        - Use full spot capacity (100%) if spot is attractive
        - Reduce spot usage if spot price is too high

        Logic:
        - If spot_price < contract_price * 0.90 (10% discount): USE ALL spot (100%)
        - If spot_price > contract_price * 1.10 (10% premium): USE MIN spot (0% - fully optional)
        - Otherwise: LINEAR interpolation between 0% and 100%

        Returns:
            Utilization factor between 0.0 and 1.0
        """
        # Check if we have an active contract to compare against
        if not self.active_contract:
            # No contract - always use all spot capacity
            return 1.0

        # Get current year
        from src.utils import year_for_tick
        current_year = year_for_tick(
            int(self.model.config["start_year"]),
            int(self.model.schedule.time)
        )

        # Get current contract price (escalated)
        contract_price = self.active_contract.get_price_for_year(current_year)

        # Get spot price for this state
        spot_price = self.model.state_spot_prices.get(
            self.state_id,
            self.aggregator.feedstock_price  # Fallback
        )

        # Decision thresholds
        attractive_threshold = contract_price * 0.90  # 10% discount
        expensive_threshold = contract_price * 1.10   # 10% premium

        # Case 1: Spot is attractive (≥10% cheaper) → Use all spot capacity
        if spot_price <= attractive_threshold:
            return 1.0

        # Case 2: Spot is expensive (≥10% more expensive) → Use minimum spot
        elif spot_price >= expensive_threshold:
            return 0.0  # Fully optional - can go to 0%

        # Case 3: In between → Linear interpolation
        else:
            # spot_price is between attractive and expensive
            # Map linearly: attractive (1.0) → expensive (0.0)
            price_range = expensive_threshold - attractive_threshold
            price_diff = spot_price - attractive_threshold
            utilization = 1.0 - 1.0 * (price_diff / price_range)
            return utilization

    def calculate_production_output(self) -> float:
        """
        Compute annual production with contract priority.

        Production splits into two components:
        1. Contracted capacity: uses contracted_load_factor (priority allocation)
        2. Spot capacity: uses spot_load_factor (residual allocation)
           - NEW: Spot capacity usage is modulated by get_spot_utilization_factor()
           - Plant reduces spot purchases when spot price is unfavorable

        Returns:
            Annual production volume (contracted + spot)
        """
        contracted_capacity = self.get_contracted_capacity()
        spot_capacity = self.get_spot_capacity()

        contracted_production = (
            contracted_capacity * self.design_load_factor *
            self.aggregator.contracted_load_factor * self.streamday_percentage
        )

        # CLAUDE NEW - Spot Capacity Optimization
        # Get utilization factor based on price signals
        spot_utilization = self.get_spot_utilization_factor()

        spot_production = (
            spot_capacity * spot_utilization *  # ← NEW: Price-responsive utilization
            self.design_load_factor *
            self.aggregator.spot_load_factor * self.streamday_percentage
        )

        return contracted_production + spot_production

 

    def update_supply(self) -> None:

        """Mesa step, production site not involved."""

        self.current_tick = self.model.schedule.time

        pass

 

    def produce(self) -> None:

        """

        Stage- produce:

        - If not yet operational, sets production to zero.

        - Else computes production based on design + realised load factors.

        """

        self.streamday_percentage = self.sample_streamday_percentage()

        current_tick: int = self.model.schedule.time

 

        if self.operational_year > current_tick:

            self.year_production_output = 0.0

        else:

            self.year_production_output = self.calculate_production_output()

 

    def evaluate(self) -> None:

        """Mesa step, production site not involved."""

        pass

 

    def invest(self) -> None:

        """Mesa step, production site not involved."""

        pass

 

    def __repr__(self) -> str:

        """

        Debug representation.

 

        Returns:

            Readable summary string.

        """

        return (

            f"SAFProductionSite(site_id='{self.site_id}', "

            f"state_id='{self.state_id}', "

            f"investor_id='{self.investor_id}', "

            f"capacity={self.max_capacity}, "

            f"DLF={self.design_load_factor:.2f}, "

            f"ALF={self.aggregator.annual_load_factor:.2f}, "

            f"SRMC={self.srmc:.2f}, "

            f"Output={self.tick_production_output:.2f})"

        )

 

    @classmethod

    def generate_site_id(cls, state_id: str) -> str:

        cls.site_counter += 1

        """

        Generate a unique site identifier.

 

        Parameters:

            state_id: State identifier to embed.

        Returns:

            Unique site ID string.

        """

        return f"site_{state_id}_{cls.site_counter:03d}"

 

