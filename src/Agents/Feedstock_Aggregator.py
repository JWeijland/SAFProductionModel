from mesa import Agent

import random

from typing import Tuple, Dict, Any, List

import logging

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.Agents.FeedstockContract import FeedstockContract

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

        # Contract tracking
        self.active_contracts: List['FeedstockContract'] = []

        # Base price for escalation calculations (kept for backward compatibility)
        self.base_feedstock_price: float = state_info["feedstock_price"]

        # Contract priority: contracted capacity gets priority over spot during shortages
        self.total_contracted_demand: float = 0.0
        self.total_spot_demand: float = 0.0
        self.contracted_load_factor: float = 1.0  # Priority allocation
        self.spot_load_factor: float = 1.0  # Residual allocation

        # TIERED PRICING SYSTEM - New implementation
        # Configuration for tier-based pricing (no escalation)
        self.tier_capacity_size: float = float(model.config.get("tier_capacity_size", 120_000))
        self.tier_1_cost: float = float(model.config.get("tier_1_cost", 400))
        self.tier_cost_increment: float = float(model.config.get("tier_cost_increment", 200))
        self.aggregator_profit_margin: float = float(model.config.get("aggregator_profit_margin", 50))
        self.spot_premium: float = float(model.config.get("spot_premium", 0.10))  # 10% premium for spot purchases

        # Generate tiers dynamically based on max_supply
        self.tiers: List[Dict[str, float]] = self._generate_tiers()

        # Track cumulative allocated capacity (ton/year) - PERMANENT
        self.cumulative_allocated: float = 0.0

        logger.info(
            f"Aggregator {self.state_id} initialized with {len(self.tiers)} tiers, "
            f"total capacity: {self.max_supply:.0f} ton/year"
        )



    def _generate_tiers(self) -> List[Dict[str, float]]:
        """
        Generate pricing tiers dynamically based on max_supply.

        Each tier represents a capacity band with increasing cost.
        Tiers continue until max_supply is reached.

        Returns:
            List of tier dictionaries with keys: 'start', 'end', 'capacity', 'cost'
        """
        tiers = []
        cumulative_capacity = 0.0
        tier_index = 0

        while cumulative_capacity < self.max_supply:
            # Calculate tier capacity (last tier may be smaller)
            tier_capacity = min(
                self.tier_capacity_size,
                self.max_supply - cumulative_capacity
            )

            # Calculate tier cost (increases linearly)
            tier_cost = self.tier_1_cost + (tier_index * self.tier_cost_increment)

            tiers.append({
                "start": cumulative_capacity,
                "end": cumulative_capacity + tier_capacity,
                "capacity": tier_capacity,
                "cost": tier_cost,
                "tier_number": tier_index + 1
            })

            cumulative_capacity += tier_capacity
            tier_index += 1

        logger.info(
            f"Generated {len(tiers)} tiers for {self.state_id}: "
            f"Tier 1 @ ${tiers[0]['cost']:.0f}/ton, "
            f"Tier {len(tiers)} @ ${tiers[-1]['cost']:.0f}/ton"
        )

        return tiers

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

        Calculate load factors with contract priority based on current_supply.

        IMPORTANT: current_supply should already be sampled in Model.step() before this is called!

        Two-tier allocation:
          1. Contracted capacity gets priority (100% load factor when supply allows)
          2. Spot capacity gets residual supply (scaled during shortages)

        Side-effects:

          - Updates contracted_load_factor, spot_load_factor
          - Maintains annual_load_factor for backward compatibility
          - Does NOT resample current_supply (already done in Model.step())

        """

        # Use the current_supply that was already sampled in Model.step()
        actual_supply = self.current_supply

        total_demand = self.total_contracted_demand + self.total_spot_demand

        # Two-tier allocation: contracted priority, then spot
        if total_demand <= actual_supply:
            # Surplus: everyone gets 100%
            self.contracted_load_factor = 1.0
            self.spot_load_factor = 1.0
            self.annual_load_factor = 1.0

        elif self.total_contracted_demand > 0:
            # Shortage: contracts have priority
            if self.total_contracted_demand <= actual_supply:
                # Enough for contracts, spot gets remainder
                self.contracted_load_factor = 1.0
                remaining_supply = actual_supply - self.total_contracted_demand
                self.spot_load_factor = (
                    remaining_supply / self.total_spot_demand
                    if self.total_spot_demand > 0 else 1.0
                )
                # Weighted average for backward compatibility
                self.annual_load_factor = (
                    (self.total_contracted_demand + self.total_spot_demand * self.spot_load_factor) /
                    total_demand if total_demand > 0 else 1.0
                )

            else:
                # Severe shortage: even contracts scale down
                self.contracted_load_factor = actual_supply / self.total_contracted_demand
                self.spot_load_factor = 0.0
                self.annual_load_factor = self.contracted_load_factor

        else:
            # No contracts: use spot allocation for all capacity
            if total_demand > 0:
                self.spot_load_factor = actual_supply / total_demand
                self.annual_load_factor = self.spot_load_factor
            else:
                self.spot_load_factor = 1.0
                self.annual_load_factor = 1.0
            self.contracted_load_factor = 1.0

 

    def produce(self) -> None:

        """Mesa step, aggregator not involved."""

        pass

 

    def evaluate(self) -> None:

        """Mesa step, aggregator not involved."""

        pass

 

    def invest(self) -> None:

        """Mesa step, aggregator not involved."""

        pass

    # ==================== TIERED PRICING METHODS ====================

    def _calculate_weighted_price(self, requested_volume: float) -> float:
        """
        Calculate weighted average price across tiers WITHOUT committing allocation.

        This method simulates what price would be charged for a given volume
        based on current cumulative_allocated position.

        Args:
            requested_volume: Annual feedstock volume needed (ton/year)

        Returns:
            Weighted average price (USD/ton) including profit margin
        """
        if requested_volume <= 0:
            return self.tier_1_cost + self.aggregator_profit_margin

        total_cost = 0.0
        remaining = requested_volume
        position = self.cumulative_allocated  # Start from current allocation

        for tier in self.tiers:
            if remaining <= 0:
                break

            # How much can we take from this tier?
            if position < tier["end"]:
                available_in_tier = tier["end"] - max(position, tier["start"])
                take_from_tier = min(remaining, available_in_tier)

                total_cost += take_from_tier * tier["cost"]
                remaining -= take_from_tier
                position += take_from_tier

        # Handle overflow (if requested exceeds max_supply)
        if remaining > 0:
            logger.warning(
                f"State {self.state_id}: Requested {requested_volume:.0f} ton/year "
                f"would exceed available capacity. Shortage: {remaining:.0f} ton/year. "
                f"Using highest tier cost for overflow."
            )
            # Use highest tier cost for overflow
            total_cost += remaining * self.tiers[-1]["cost"]

        # Calculate weighted average + absolute profit margin
        weighted_avg = total_cost / requested_volume
        final_price = weighted_avg + self.aggregator_profit_margin

        return final_price

    def allocate_contract(self, annual_volume: float, current_year: int) -> float:
        """
        Calculate price and PERMANENTLY allocate capacity for a new contract.

        This method commits the capacity by increasing cumulative_allocated.
        The allocated capacity remains reserved even after contract expiry
        (plant continues operating, may renew at same price).

        Args:
            annual_volume: Annual feedstock volume (ton/year) to allocate
            current_year: Year of contract signing

        Returns:
            Contract price (USD/ton) - fixed for contract duration (NO escalation)
        """
        # Calculate price based on current tier position
        contract_price = self._calculate_weighted_price(annual_volume)

        # Permanently allocate capacity
        old_allocated = self.cumulative_allocated
        self.cumulative_allocated += annual_volume

        # Determine which tiers were used
        tiers_used = []
        position = old_allocated
        remaining = annual_volume

        for tier in self.tiers:
            if remaining <= 0:
                break
            if position < tier["end"]:
                available = tier["end"] - max(position, tier["start"])
                taken = min(remaining, available)
                if taken > 0:
                    tiers_used.append(f"Tier {tier['tier_number']} ({taken:.0f} ton @ ${tier['cost']:.0f}/ton)")
                remaining -= taken
                position += taken

        logger.info(
            f"State {self.state_id} - Contract allocated in year {current_year}:"
        )
        logger.info(
            f"  Volume: {annual_volume:.0f} ton/year @ ${contract_price:.2f}/ton (weighted)"
        )
        logger.info(
            f"  Tiers used: {', '.join(tiers_used)}"
        )
        logger.info(
            f"  Cumulative allocated: {old_allocated:.0f} → {self.cumulative_allocated:.0f} ton/year "
            f"({self.cumulative_allocated/self.max_supply*100:.1f}% of max supply)"
        )

        return contract_price

    def renew_contract_at_same_tier(self, existing_contract: 'FeedstockContract', current_year: int) -> float:
        """
        Renew expired contract at the SAME tier price without re-allocating capacity.

        Capacity was already permanently allocated during initial contract.
        Renewal maintains the same tier pricing without moving cumulative_allocated.

        This prevents double-counting of capacity and ensures plants don't pay
        higher tier prices after renewal (they keep their original tier position).

        Args:
            existing_contract: The expired contract to renew
            current_year: Year of renewal

        Returns:
            Renewal price (same as original contract price - NO tier escalation)
        """
        # Use the SAME price as the original contract
        # No escalation in tier system, so renewal = original price
        renewal_price = existing_contract.initial_contract_price

        logger.info(
            f"State {self.state_id} - Contract RENEWED in year {current_year}:"
        )
        logger.info(
            f"  Volume: {existing_contract.contracted_volume:.0f} ton/year "
            f"@ ${renewal_price:.2f}/ton (SAME tier as original)"
        )
        logger.info(
            f"  Cumulative allocated: {self.cumulative_allocated:.0f} ton/year (UNCHANGED - no double counting)"
        )
        logger.info(
            f"  Original contract: year {existing_contract.start_year}-{existing_contract.end_year}"
        )

        return renewal_price

    def get_spot_price(self, spot_volume: float) -> float:
        """
        Calculate spot market price WITHOUT permanent allocation.

        Spot market uses same tier logic but doesn't commit capacity.
        Good for one-time purchases or spot market transactions.

        Args:
            spot_volume: Volume to price (ton/year)

        Returns:
            Spot price (USD/ton)
        """
        return self._calculate_weighted_price(spot_volume)

    def get_marginal_feedstock_price(self) -> float:
        """
        Get MARGINAL feedstock cost - the cost of the NEXT tonne of feedstock.

        This is used for merit order pricing to determine the short-run marginal cost.
        Returns the tier price at the current cumulative_allocated position.

        Returns:
            Marginal feedstock price (USD/ton) including aggregator margin
        """
        # Find which tier we're currently in
        position = self.cumulative_allocated

        for tier in self.tiers:
            if position < tier["end"]:
                # We're in this tier - return its cost + margin
                return tier["cost"] + self.aggregator_profit_margin

        # If we're beyond all tiers (shouldn't happen), return highest tier cost
        return self.tiers[-1]["cost"] + self.aggregator_profit_margin

    def get_current_market_price(self, current_year: int) -> float:
        """
        Get current market feedstock price (for backward compatibility).

        NEW BEHAVIOR: Returns tier-based price for a typical plant capacity.
        NO ESCALATION - price is purely tier-based.

        This method is kept for backward compatibility but now uses
        the tiered pricing system.

        Args:
            current_year: Current year (not used in tier system, kept for API compatibility)

        Returns:
            Market feedstock price (USD/ton)
        """
        # Return price for a typical contract volume (e.g., 85,000 ton/year)
        typical_volume = 85_000
        return self._calculate_weighted_price(typical_volume)

    def get_spot_price(self, current_year: int = None) -> float:
        """
        Get current spot market price for non-contracted feedstock.

        Spot price is based on the CURRENT tier position (marginal cost) + spot premium.
        This reflects the price buyers pay for feedstock without long-term commitment.

        Logic:
        - Get the marginal tier price at current cumulative_allocated position
        - Add spot premium (default 10%) for no commitment risk

        Args:
            current_year: Current year (not used, kept for API compatibility)

        Returns:
            Spot market price (USD/tonne)

        Example:
            If cumulative_allocated is at Tier 2 (600 USD/tonne) with 10% premium:
            spot_price = 600 * 1.10 = 660 USD/tonne
        """
        try:
            # Get the marginal tier price (already includes aggregator margin)
            marginal_cost = self.get_marginal_feedstock_cost()

            # Add spot premium for no long-term commitment
            spot_price = marginal_cost * (1 + self.spot_premium)

            logger.debug(
                f"State {self.state_id}: Spot price = ${marginal_cost:.0f} "
                f"(tier) * {1+self.spot_premium:.2f} (premium) = ${spot_price:.0f}/tonne"
            )

            return spot_price
        except Exception as e:
            logger.error(
                f"Error calculating spot price for {self.state_id}: {e}. "
                f"Using fallback tier 1 cost."
            )
            # Fallback to tier 1 cost + margin + premium
            fallback = (self.tier_1_cost + self.aggregator_profit_margin) * (1 + self.spot_premium)
            return fallback

    def register_contract(self, contract: 'FeedstockContract') -> None:
        """Register a new feedstock contract with this aggregator."""
        self.active_contracts.append(contract)

        logging.info(
            f"Aggregator {self.state_id} registered contract {contract.contract_id}: "
            f"{contract.contracted_volume:.0f} tonnes/year for {contract.duration} years"
        )

    def get_contracted_capacity(self, current_year: int) -> float:
        """Calculate total capacity committed via active contracts in given year."""
        total_contracted = sum(
            contract.contracted_volume
            for contract in self.active_contracts
            if contract.is_active(current_year)
        )

        return total_contracted

    def get_available_capacity(self, current_year: int) -> float:
        """Calculate uncontracted capacity available for new investments."""
        contracted = self.get_contracted_capacity(current_year)
        available = self.max_supply - contracted

        logging.debug(
            f"Aggregator {self.state_id} year {current_year}: "
            f"max={self.max_supply:.0f}, contracted={contracted:.0f}, "
            f"available={available:.0f} tonnes/year"
        )

        return max(0, available)



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

 

 

 