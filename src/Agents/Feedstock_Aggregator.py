from mesa import Agent

import random

from typing import Tuple, Dict, Any, List

import logging

# CLAUDE START - Import for Phase 1 contract implementation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.Agents.FeedstockContract import FeedstockContract
# CLAUDE END - Import for Phase 1 contract implementation

 

logger = logging.getLogger("Feedstock Aggregator")

 

class FeedstockAggregator(Agent):

    """

    Agent representing feedstock market conditions for a single state.

 

    Responsibilities:

      - Holds max theoretical feedstock supply and feedstock price.

      - Samples stochastic realisable supply each tick via a multiplicative factor.

      - Provides annual_load_factor used by production sites (proxy for feedstock utilisation).

 

    Attributes:

        state_id (str): State identifier.

        max_supply (float): Maximum theoretical annual feedstock supply.

        feedstock_price (float): Feedstock price (per unit).

        multiplier_min (float): Lower bound for stochastic supply multiplier.

        multiplier_max (float): Upper bound for stochastic supply multiplier.

        current_supply (float): Sampled supply this tick based on variable feedstock availability.

        annual_load_factor (float): Multiplier to find sampled annual load.

    """

 

    def __init__(

        self,

        unique_id: Any,

        model: Any,

        state_id: str,

        states_data: Dict[str, Dict[str, float]],

    ):

        super().__init__(unique_id, model)

        """

        Initialise the Feedstock Aggregator agent with state-specific data and configuration.

 

        Validation:

          - Ensures state_id exists in states_data.

          - Validates numeric fields are non-negative.

 

        Args:

            unique_id: Unique identifier for the agent.

            model: SAF Market Model.

            state_id: State identifier.

            states_data: Mapping of state_id -> dict.

        """

        if not isinstance(state_id, str):

            raise TypeError("state_id must be a string.")

        if state_id not in states_data:

            raise ValueError(f"No feedstock price found for state '{state_id}'.")

 

        state_info = states_data[state_id]

        self.state_id: str = state_id

        self.max_supply: float = state_info["max_supply"]

        self.feedstock_price: float = state_info["feedstock_price"]

        self.multiplier_min: float = model.config["feedstock_multiplier_min"]

        self.multiplier_max: float = model.config["feedstock_multiplier_max"]

        self.current_supply, self.annual_load_factor = self.sample_current_supply()

 

        if not isinstance(self.feedstock_price, (int, float)):

            raise TypeError("feedstock_price must be a number.")

        if not isinstance(self.max_supply, (int, float)):

            raise TypeError("max_supply must be a number.")

        if self.max_supply < 0:

            raise ValueError("max_supply must be non-negative.")

        if self.feedstock_price < 0:

            raise ValueError("feedstock_price must be non-negative.")

        # CLAUDE START - Contract tracking for Phase 1 implementation
        self.active_contracts: List['FeedstockContract'] = []
        # CLAUDE END - Contract tracking for Phase 1 implementation



    def sample_current_supply(self) -> Tuple[float, float]:

        """

        Sample current feedstock availability.

 

        Logic:

          - Draw random multiplier (multiplier_min, multiplier_max).

          - Sampled supply = max_supply * multiplier (capped at max_supply).

          - annual_load_factor = min(1, multiplier) for plant load approximation.

 

        Returns:

            (current_supply, annual_load_factor)

        """

        multiplier = random.uniform(self.multiplier_min, self.multiplier_max)

        return min(self.max_supply, self.max_supply * multiplier), min(1, multiplier)

 

    def update_supply(self) -> None:

        """

        Resample current supply and annual load factor for this tick.

 

        Side-effects:

          - Updates current_supply and annual_load_factor attributes.

          - Logs data.

        """

        if self.available_feedstock < 1:

            self.current_supply, self.annual_load_factor = self.sample_current_supply()

        else:

            self.current_supply = self.max_supply

            self.annual_load_factor = 1.0

 

        logging.info(

            f"Annual load factor of {self.state_id} is: {self.annual_load_factor:.2f}"

        )

 

    def produce(self) -> None:

        """Mesa step, aggregator not involved."""

        pass

 

    def evaluate(self) -> None:

        """Mesa step, aggregator not involved."""

        pass

 

    def invest(self) -> None:

        """Mesa step, aggregator not involved."""

        pass

    # CLAUDE START - Contract management methods for Phase 1 implementation
    def register_contract(self, contract: 'FeedstockContract') -> None:
        """
        Register a new contract with this aggregator.

        Called by model when an investor signs a contract for feedstock
        from this state.

        Parameters:
            contract: FeedstockContract to register

        Side effects:
            - Adds contract to active_contracts list
        """
        self.active_contracts.append(contract)

        logging.info(
            f"Aggregator {self.state_id} registered contract {contract.contract_id}: "
            f"{contract.contracted_volume:.0f} tonnes/year for {contract.duration} years"
        )

    def get_contracted_capacity(self, current_year: int) -> float:
        """
        Calculate total capacity committed via active contracts.

        This is the amount of feedstock that is already pledged to
        existing contracts in the given year.

        Parameters:
            current_year: Year to check contracts for

        Returns:
            Total contracted volume in tonnes/year
        """
        total_contracted = sum(
            contract.contracted_volume
            for contract in self.active_contracts
            if contract.is_active(current_year)
        )

        return total_contracted

    def get_available_capacity(self, current_year: int) -> float:
        """
        Calculate uncontracted capacity available for new investments.

        This is max_supply minus already contracted capacity.

        Parameters:
            current_year: Year to check availability for

        Returns:
            Available capacity in tonnes/year

        Note:
            Result can be negative if contracts exceed max_supply
            (shouldn't happen in normal operation)
        """
        contracted = self.get_contracted_capacity(current_year)
        available = self.max_supply - contracted

        logging.debug(
            f"Aggregator {self.state_id} year {current_year}: "
            f"max={self.max_supply:.0f}, contracted={contracted:.0f}, "
            f"available={available:.0f} tonnes/year"
        )

        return max(0, available)
    # CLAUDE END - Contract management methods for Phase 1 implementation



    def __repr__(self) -> str:

        """

        Debug representation of aggregator state.

 

        Returns:

            Readable summary string.

        """

        return (

            f"FeedstockAggregator(state_id='{self.state_id}', "

            f"max_supply={self.max_supply}, "

            f"current_supply={self.current_supply:.2f}, "

            f"feedstock_price={self.feedstock_price})"

        )

 

 

 