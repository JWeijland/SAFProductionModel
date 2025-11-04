from src.Agents.Feedstock_Aggregator import FeedstockAggregator

from mesa import Agent

from typing import List, Any

import logging

import random

 

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

 

    def calculate_srmc(self) -> float:

        """

        Compute short-run marginal cost (SRMC): Cost to produce 1 tonne of SAF.

 

        Formula:

          SRMC = feedstock_price + opex + transport_cost + profit_margin

 

        Returns:

            SRMC value (float).

        """

        feedstock_price = self.aggregator.feedstock_price

 

        return feedstock_price + self.opex + self.transport_cost + self.profit_margin

 

    def calculate_production_output(self) -> float:

        """

        Compute potential annual production given current aggregator load factor.

 

        Returns:

            Annual production volume (float).

        """

        return (

            self.max_capacity

            * self.design_load_factor

            * self.aggregator.annual_load_factor

            * self.streamday_percentage

        )

 

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

 

