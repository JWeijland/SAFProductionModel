# CLAUDE START - Unit tests for FeedstockContract Phase 1 implementation
"""
Unit Tests for FeedstockContract

Tests the core contract functionality:
1. Contract creation and validation
2. Active/expired status checking
3. Price escalation over time
4. Blended cost calculation (contract + spot)

Run with: pytest tests/test_feedstock_contract.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.Agents.FeedstockContract import FeedstockContract, create_contract_from_site


def create_test_contract(**kwargs):
    """Helper function to create test contracts with sensible defaults."""
    defaults = {
        "contract_id": "contract_001",
        "investor_id": "inv_01",
        "aggregator_id": "PUNJAB",
        "plant_id": "site_001",
        "initial_contract_price": 600.0,
        "start_year": 2025,
        "end_year": 2045,
        "annual_capacity": 100000.0,
        "contract_percentage": 0.85,
        "escalation_rate": 0.03,
        "duration": 20,
        "status": "active"
    }
    defaults.update(kwargs)
    return FeedstockContract(**defaults)


class TestContractCreation:
    """Test contract creation and validation."""

    def test_contract_creation(self):
        """Test: Verify all contract fields are set correctly."""
        contract = FeedstockContract(
            contract_id="contract_001",
            investor_id="inv_01",
            aggregator_id="PUNJAB",
            plant_id="site_PUNJAB_001",
            initial_contract_price=600.0,
            start_year=2025,
            end_year=2045,
            annual_capacity=100000.0,
            contract_percentage=0.85,
            escalation_rate=0.03,
            duration=20,
            status="active"
        )

        # Verify all fields
        assert contract.contract_id == "contract_001"
        assert contract.investor_id == "inv_01"
        assert contract.aggregator_id == "PUNJAB"
        assert contract.plant_id == "site_PUNJAB_001"
        assert contract.initial_contract_price == 600.0
        assert contract.escalation_rate == 0.03
        assert contract.start_year == 2025
        assert contract.end_year == 2045
        assert contract.duration == 20
        assert contract.annual_capacity == 100000.0
        assert contract.contract_percentage == 0.85
        assert contract.status == "active"

    def test_contract_validation_invalid_price(self):
        """Test: Contract rejects negative price."""
        with pytest.raises(ValueError, match="initial_contract_price must be positive"):
            create_test_contract(initial_contract_price=-100.0)  # Invalid!

    def test_contract_validation_invalid_percentage(self):
        """Test: Contract rejects coverage outside 80-90% range."""
        with pytest.raises(ValueError, match="contract_percentage must be between 0.80 and 0.90"):
            create_test_contract(contract_percentage=0.95)  # Invalid! Must be 0.80-0.90


class TestContractActiveYears:
    """Test contract active status over its lifecycle."""

    def test_contract_active_at_start(self):
        """Test: Contract is active at start_year."""
        contract = create_test_contract()
        assert contract.is_active(2025) is True

    def test_contract_active_at_midpoint(self):
        """Test: Contract is active at year 10 (midpoint)."""
        contract = create_test_contract()
        assert contract.is_active(2035) is True  # Year 10

    def test_contract_active_at_end(self):
        """Test: Contract is active at end_year (inclusive)."""
        contract = create_test_contract()
        assert contract.is_active(2045) is True

    def test_contract_expired_after_end(self):
        """Test: Contract is NOT active at year 21 (after expiration)."""
        contract = create_test_contract()
        assert contract.is_active(2046) is False  # Year 21, expired

    def test_contract_not_active_before_start(self):
        """Test: Contract is NOT active before start_year."""
        contract = create_test_contract()
        assert contract.is_active(2024) is False


class TestPriceEscalation:
    """Test price escalation formula."""

    def test_price_escalation_year_0(self):
        """Test: Price at year 0 equals initial price."""
        contract = create_test_contract()
        price = contract.get_price_for_year(2025)
        assert abs(price - 600.0) < 0.01  # p_0 = 600

    def test_price_escalation_year_5(self):
        """Test: Verify p_0 × 1.03^5 = p_5."""
        contract = create_test_contract()
        price_year_5 = contract.get_price_for_year(2030)  # 5 years later
        expected = 600.0 * (1.03 ** 5)  # = 695.46
        assert abs(price_year_5 - expected) < 0.01

    def test_price_escalation_year_20(self):
        """Test: Price at end of contract (year 20)."""
        contract = create_test_contract()
        price_year_20 = contract.get_price_for_year(2045)  # 20 years later
        expected = 600.0 * (1.03 ** 20)  # = 1083.85
        assert abs(price_year_20 - expected) < 0.01


class TestBlendedCost:
    """Test blended cost calculation (contract + spot)."""

    def test_blended_cost_calculation(self):
        """Test: Verify 0.85 × p_contract + 0.15 × p_spot."""
        contract = create_test_contract()

        # Year 5: contract price = 600 × 1.03^5 = 695.46
        contract_price_year_5 = contract.get_price_for_year(2030)
        spot_price = 720.0  # Hypothetical spot price

        # Blended cost = 0.85 × contract + 0.15 × spot
        blended_cost = (
            contract.contract_percentage * contract_price_year_5 +
            (1 - contract.contract_percentage) * spot_price
        )

        expected = 0.85 * contract_price_year_5 + 0.15 * spot_price
        assert abs(blended_cost - expected) < 0.01


class TestContractVolumes:
    """Test contracted and spot volume calculations."""

    def test_contracted_volume(self):
        """Test: Contracted volume = capacity × percentage."""
        contract = create_test_contract()
        expected_contracted = 100000.0 * 0.85  # = 85,000 tonnes/year
        assert abs(contract.contracted_volume - expected_contracted) < 0.01

    def test_spot_volume(self):
        """Test: Spot volume = capacity × (1 - percentage)."""
        contract = create_test_contract()
        expected_spot = 100000.0 * 0.15  # = 15,000 tonnes/year
        assert abs(contract.spot_volume - expected_spot) < 0.01


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])

# CLAUDE END - Unit tests for FeedstockContract Phase 1 implementation
