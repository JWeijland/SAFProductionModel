# CLAUDE START - Complete FeedstockContract module for Phase 1 implementation
"""
FeedstockContract Data Structure

This module defines the contract between a feedstock aggregator and an investor's
production plant. Contracts guarantee feedstock supply at a fixed price (with annual
escalation) for a 20-year duration, covering 80-90% of plant capacity.

Phase 1 Implementation - Simplified contract system without negotiations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FeedstockContract:
    """
    Represents a long-term feedstock supply contract between an aggregator
    and an investor's production plant.

    Contract Structure:
    - Duration: 20 years (configurable)
    - Coverage: 50-100% of effective plant capacity (investor chooses, default 80-90%)
    - Effective capacity includes: design load factor × contracted load factor × stream days
    - Pricing: Initial market price with 3% annual escalation
    - Remaining percentage: Purchased at annual spot price

    Attributes:
        contract_id: Unique contract identifier
        investor_id: ID of investor holding contract
        aggregator_id: ID of aggregator (state_id)
        plant_id: ID of production plant covered by contract
        initial_contract_price: SRMC at contract signing (USD/tonne)
        escalation_rate: Annual price increase rate (default: 0.03)
        start_year: Year contract begins
        end_year: Year contract expires (start_year + duration)
        duration: Contract length in years (default: 20)
        annual_capacity: Plant's effective maximum capacity (tonnes/year)
            Calculated as: max_capacity × design_load_factor × contracted_load_factor × stream_days
        contract_percentage: Fraction covered by contract (0.50-1.00, typically 0.80-0.90)
        status: Contract state ("active" or "expired")
    """

    # Required fields (no defaults) must come first
    contract_id: str
    investor_id: str
    aggregator_id: str  # state_id
    plant_id: str
    initial_contract_price: float  # USD/tonne, SRMC at signing
    start_year: int
    end_year: int  # start_year + duration
    annual_capacity: float  # Plant's max capacity (tonnes/yr)
    contract_percentage: float  # 0.80 to 0.90

    # Optional fields (with defaults) must come after
    escalation_rate: float = 0.03  # Annual increase (3% default)
    duration: int = 20  # years
    status: str = "active"  # "active" or "expired"

    def __post_init__(self):
        """Validate contract parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate contract parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate pricing
        if self.initial_contract_price <= 0:
            raise ValueError(
                f"initial_contract_price must be positive, got {self.initial_contract_price}"
            )

        if not 0 <= self.escalation_rate <= 0.10:
            raise ValueError(
                f"escalation_rate must be between 0 and 0.10 (0-10%), got {self.escalation_rate}"
            )

        # Validate timing
        if self.duration <= 0:
            raise ValueError(
                f"duration must be positive, got {self.duration}"
            )

        if self.end_year != self.start_year + self.duration:
            raise ValueError(
                f"end_year ({self.end_year}) must equal start_year ({self.start_year}) "
                f"+ duration ({self.duration})"
            )

        # Validate capacity
        if self.annual_capacity <= 0:
            raise ValueError(
                f"annual_capacity must be positive, got {self.annual_capacity}"
            )

        if not 0.50 <= self.contract_percentage <= 1.00:
            raise ValueError(
                f"contract_percentage must be between 0.50 and 1.00, got {self.contract_percentage}"
            )

        # Validate status
        if self.status not in ["active", "expired"]:
            raise ValueError(
                f"status must be 'active' or 'expired', got {self.status}"
            )

    @property
    def contracted_volume(self) -> float:
        """
        Calculate annual volume guaranteed by contract.

        Returns:
            Annual contracted volume in tonnes/year
        """
        return self.annual_capacity * self.contract_percentage

    @property
    def spot_volume(self) -> float:
        """
        Calculate annual volume purchased on spot market.

        Returns:
            Annual spot volume in tonnes/year (10-20% of capacity)
        """
        return self.annual_capacity * (1 - self.contract_percentage)

    def get_price_for_year(self, current_year: int) -> float:
        """
        Get escalated contract price for a specific year.

        Formula: p(t) = p_0 × (1 + r)^t
        where:
            p_0 = initial_contract_price
            r = escalation_rate
            t = years since contract start

        Args:
            current_year: Year to calculate price for

        Returns:
            Escalated contract price in USD/tonne

        Raises:
            ValueError: If current_year is before contract start
        """
        if current_year < self.start_year:
            raise ValueError(
                f"current_year ({current_year}) cannot be before "
                f"contract start_year ({self.start_year})"
            )

        years_elapsed = current_year - self.start_year
        escalated_price = self.initial_contract_price * (
            (1 + self.escalation_rate) ** years_elapsed
        )
        return escalated_price

    def is_active(self, current_year: int) -> bool:
        """
        Check if contract is currently active.

        Args:
            current_year: Year to check

        Returns:
            True if contract is active in the given year, False otherwise
        """
        return (
            self.start_year <= current_year <= self.end_year
            and self.status == "active"
        )

    def expire(self) -> None:
        """
        Mark contract as expired.

        This should be called when the contract reaches its end_year.
        """
        self.status = "expired"

    def __repr__(self) -> str:
        """
        String representation for debugging.

        Returns:
            Readable summary of contract
        """
        return (
            f"FeedstockContract("
            f"id='{self.contract_id}', "
            f"investor='{self.investor_id}', "
            f"aggregator='{self.aggregator_id}', "
            f"plant='{self.plant_id}', "
            f"price=${self.initial_contract_price:.2f}/tonne, "
            f"escalation={self.escalation_rate:.1%}, "
            f"years={self.start_year}-{self.end_year}, "
            f"coverage={self.contract_percentage:.1%}, "
            f"status='{self.status}'"
            f")"
        )


def create_contract_from_site(
    site,
    investor_id: str,
    current_year: int,
    contract_percentage: float,
    duration: int = 20,
    escalation_rate: float = 0.03
) -> FeedstockContract:
    """
    Factory function to create a contract from a SAFProductionSite.

    This is a convenience function to create contracts during the
    investment phase.

    Args:
        site: SAFProductionSite instance
        investor_id: ID of investor creating contract
        current_year: Year contract is signed
        contract_percentage: Coverage fraction (0.50-1.00, typically 0.80-0.90)
        duration: Contract length in years (default: 20)
        escalation_rate: Annual price escalation (default: 0.03)

    Returns:
        New FeedstockContract instance
    """
    contract = FeedstockContract(
        contract_id=f"contract_{site.site_id}",
        investor_id=investor_id,
        aggregator_id=site.state_id,
        plant_id=site.site_id,
        initial_contract_price=site.srmc,  # Use site's SRMC as base price
        escalation_rate=escalation_rate,
        start_year=current_year,
        end_year=current_year + duration,
        duration=duration,
        annual_capacity=site.max_capacity * site.design_load_factor,
        contract_percentage=contract_percentage,
        status="active"
    )
    return contract


# CLAUDE END - Complete FeedstockContract module for Phase 1 implementation
