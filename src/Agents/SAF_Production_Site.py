from src.Agents.Feedstock_Aggregator import FeedstockAggregator

from mesa import Agent

from typing import List, Any, Optional, TYPE_CHECKING

import logging

import random

from src.utils import year_for_tick

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

 

    def calculate_srmc(self, current_year: int = None, use_marginal_cost: bool = False) -> float:
        """
        Compute short-run marginal cost (SRMC): Cost to produce 1 tonne of SAF.

        Formula: SRMC = feedstock_price + opex + transport_cost + profit_margin

        If current_year provided, all cost components escalate at market_escalation_rate.
        Otherwise uses base prices (backwards compatibility).

        Parameters:
            current_year: Optional calendar year for market price escalation
            use_marginal_cost: If True, use marginal feedstock cost (for merit order).
                             If False, use weighted average over typical volume (for contracts/NPV).

        Returns:
            SRMC value (float).
        """
        if current_year is not None:
            start_year = int(self.model.config["start_year"])
            years_elapsed = current_year - start_year
            market_escalation_rate = float(self.model.config.get("market_escalation_rate", 0.02))
            escalation_factor = (1 + market_escalation_rate) ** years_elapsed

            if use_marginal_cost:
                # Merit order: use marginal cost (cost of next tonne at current tier position)
                feedstock_price = self.aggregator.get_marginal_feedstock_price()
            else:
                # Contracts/NPV: use weighted average over typical volume
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
        Decide how much of spot capacity to utilize.

        SPOT UTILIZATION: DISABLED
        Plants always use 100% of their spot capacity.

        Rationale:
        - If there is market demand, plants should produce to meet it
        - Spot feedstock cost is already included in SRMC for merit order
        - Demand allocation handles oversupply situations
        - Price optimization happens at investment stage, not production stage

        Returns:
            Always returns 1.0 (100% utilization)
        """
        # SPOT OPTIMIZATION DISABLED - Always use full spot capacity
        return 1.0

    def calculate_production_output(self) -> float:
        """
        Compute annual production with contract priority.

        Production splits into two components:
        1. Contracted capacity: uses contracted_load_factor (priority allocation)
        2. Spot capacity: uses spot_load_factor (residual allocation)
           - Spot utilization is always 100% (optimization disabled)
           - Plants produce to full capacity when feedstock is available

        Returns:
            Annual production volume (contracted + spot)
        """
        contracted_capacity = self.get_contracted_capacity()
        spot_capacity = self.get_spot_capacity()

        contracted_production = (
            contracted_capacity * self.design_load_factor *
            self.aggregator.contracted_load_factor * self.streamday_percentage
        )

        # Spot utilization factor (always 1.0 - optimization disabled)
        spot_utilization = self.get_spot_utilization_factor()

        spot_production = (
            spot_capacity * spot_utilization *  # Always 1.0 (100%)
            self.design_load_factor *
            self.aggregator.spot_load_factor * self.streamday_percentage
        )

        return contracted_production + spot_production

    def calculate_take_or_pay_penalty(
        self,
        allocated_production: float,
        current_year: int
    ) -> float:
        """
        Calculate penalty for not taking contracted feedstock due to demand curtailment.

        When market oversupply forces production curtailment, sites with contracts
        must pay a penalty for feedstock they committed to buy but cannot use.

        Logic:
        - Calculate expected contracted production (what contract requires)
        - Calculate actual contracted production (limited by allocation)
        - Curtailed amount = expected - actual
        - Penalty = curtailed × penalty_rate

        This implements a take-or-pay mechanism where:
        - Sites pay for contracted feedstock even if they can't use it
        - Penalty is less than full production cost (e.g., 300/tonne vs 1500/tonne SRMC)
        - Prevents sites from overcommitting to contracts

        Parameters:
            allocated_production: How much this site is allowed to produce (from demand allocation)
            current_year: Current calendar year

        Returns:
            penalty_cost: USD (positive = cost, goes to aggregator)
        """
        contracted_capacity = self.get_contracted_capacity(current_year)

        if contracted_capacity <= 0:
            # No contract, no penalty
            return 0.0

        # Calculate expected contracted production (what we SHOULD produce per contract)
        contracted_production_expected = (
            contracted_capacity *
            self.design_load_factor *
            self.aggregator.contracted_load_factor *
            self.streamday_percentage
        )

        # Calculate actual contracted production (limited by allocation)
        # We produce contracted volumes first (priority), so:
        contracted_production_actual = min(allocated_production, contracted_production_expected)

        # Curtailed contracted production (feedstock we must pay for but can't use)
        curtailed_contracted = max(0, contracted_production_expected - contracted_production_actual)

        self.curtailed_volume = curtailed_contracted

        if curtailed_contracted <= 0:
            # No curtailment, no penalty
            return 0.0

        # Penalty rate from config (default: 300 USD/tonne)
        penalty_rate = float(self.model.config.get('take_or_pay_penalty_rate', 300.0))

        penalty_cost = curtailed_contracted * penalty_rate

        # Log for transparency
        logger.info(
            f"Take-or-pay penalty for {self.site_id}: "
            f"curtailed {curtailed_contracted:,.0f} tonnes contracted feedstock, "
            f"penalty ${penalty_cost:,.0f} @ ${penalty_rate:.0f}/tonne"
        )

        return penalty_cost



    def update_supply(self) -> None:

        """Mesa step, production site not involved."""

        self.current_tick = self.model.schedule.time

        pass

 

    def produce(self) -> None:
        """
        Stage - produce:

        - If not yet operational, sets production to zero.
        - Calculates potential production based on design + realised load factors.
        - Respects demand allocation (if oversupply) to prevent overproduction.
        - Calculates take-or-pay penalty for curtailed contracted feedstock.
        """

        self.streamday_percentage = self.sample_streamday_percentage()

        current_tick: int = self.model.schedule.time
        current_year = year_for_tick(self.model.config["start_year"], current_tick)

        if self.operational_year > current_tick:
            # Not yet operational
            self.year_production_output = 0.0
            self.potential_production_output = 0.0
            self.take_or_pay_penalty = 0.0
            self.curtailed_volume = 0.0
            self.contracted_production = 0.0
            self.spot_production = 0.0
            return

        # Calculate potential production components (contracted vs spot)
        contracted_capacity = self.get_contracted_capacity(current_year)
        spot_capacity = self.get_spot_capacity(current_year)

        # Calculate potential contracted production
        potential_contracted = (
            contracted_capacity * self.design_load_factor *
            self.aggregator.contracted_load_factor * self.streamday_percentage
        )

        # Calculate potential spot production
        spot_utilization = self.get_spot_utilization_factor()
        potential_spot = (
            spot_capacity * spot_utilization *
            self.design_load_factor *
            self.aggregator.spot_load_factor * self.streamday_percentage
        )

        # Total potential production
        potential_production = potential_contracted + potential_spot
        self.potential_production_output = potential_production

        # Check if demand allocation exists (oversupply situation)
        if hasattr(self.model, 'demand_allocation') and self.model.demand_allocation is not None:
            # OVERSUPPLY: Respect allocated demand
            allocated_production = self.model.demand_allocation.get(
                self.site_id,
                potential_production  # Fallback if not in allocation dict
            )

            # Cap production at allocated amount
            actual_production = min(potential_production, allocated_production)

            # Calculate actual contracted vs spot production (proportional curtailment)
            if potential_production > 0:
                curtailment_ratio = actual_production / potential_production
                self.contracted_production = potential_contracted * curtailment_ratio
                self.spot_production = potential_spot * curtailment_ratio
            else:
                self.contracted_production = 0.0
                self.spot_production = 0.0

            # Calculate penalty for curtailed contracted production
            self.take_or_pay_penalty = self.calculate_take_or_pay_penalty(
                allocated_production,
                current_year
            )

            if actual_production < potential_production:
                logger.info(
                    f"{self.site_id} production curtailed: "
                    f"potential={potential_production:,.0f}, "
                    f"actual={actual_production:,.0f}, "
                    f"penalty=${self.take_or_pay_penalty:,.0f}"
                )
        else:
            # No allocation (supply <= demand), produce freely
            actual_production = potential_production
            self.contracted_production = potential_contracted
            self.spot_production = potential_spot
            self.take_or_pay_penalty = 0.0
            self.curtailed_volume = 0.0

        self.year_production_output = actual_production

 

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

 

