# CLAUDE START - Unit tests for contract pricing functions in utils.py
"""
Unit Tests for Contract Pricing Functions

Tests the pricing utility functions added in Phase 1:
1. calculate_initial_contract_price()
2. calculate_state_spot_price()
3. get_contract_price_for_year()

Run with: pytest tests/test_contract_pricing.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    calculate_initial_contract_price,
    calculate_state_spot_price,
    get_contract_price_for_year
)
from src.Agents.FeedstockContract import FeedstockContract


class TestCalculateInitialContractPrice:
    """Test initial contract price calculation from site SRMC."""

    def test_returns_site_srmc(self):
        """Test: Contract price equals site SRMC."""
        # Mock a site with known SRMC
        mock_site = Mock()
        mock_site.srmc = 600.0

        price = calculate_initial_contract_price(mock_site)
        assert price == 600.0

    def test_different_srmc_values(self):
        """Test: Function works with various SRMC values."""
        test_cases = [450.0, 600.0, 750.0, 1000.0]

        for expected_srmc in test_cases:
            mock_site = Mock()
            mock_site.srmc = expected_srmc

            price = calculate_initial_contract_price(mock_site)
            assert price == expected_srmc


class TestCalculateStateSpotPrice:
    """Test state-specific spot price calculation."""

    def test_average_of_new_contracts(self):
        """Test: Spot price = average of new contract prices in state."""
        # Create contracts for PUNJAB
        contracts = [
            FeedstockContract(
                contract_id="c1",
                investor_id="inv_01",
                aggregator_id="PUNJAB",
                plant_id="site_001",
                initial_contract_price=600.0,
                start_year=2025,
                end_year=2045,
                annual_capacity=100000.0,
                contract_percentage=0.85
            ),
            FeedstockContract(
                contract_id="c2",
                investor_id="inv_02",
                aggregator_id="PUNJAB",
                plant_id="site_002",
                initial_contract_price=620.0,
                start_year=2025,
                end_year=2045,
                annual_capacity=100000.0,
                contract_percentage=0.85
            ),
        ]

        spot_price = calculate_state_spot_price("PUNJAB", contracts)

        expected = (600.0 + 620.0) / 2  # = 610.0
        assert abs(spot_price - expected) < 0.01

    def test_ignores_other_states(self):
        """Test: Only counts contracts for specified state."""
        contracts = [
            FeedstockContract(
                contract_id="c1",
                investor_id="inv_01",
                aggregator_id="PUNJAB",
                plant_id="site_001",
                initial_contract_price=600.0,
                start_year=2025,
                end_year=2045,
                annual_capacity=100000.0,
                contract_percentage=0.85
            ),
            FeedstockContract(
                contract_id="c2",
                investor_id="inv_02",
                aggregator_id="MAHARASHTRA",  # Different state!
                plant_id="site_002",
                initial_contract_price=550.0,
                start_year=2025,
                end_year=2045,
                annual_capacity=100000.0,
                contract_percentage=0.85
            ),
        ]

        spot_price = calculate_state_spot_price("PUNJAB", contracts)

        # Should only use PUNJAB contract (600.0)
        assert abs(spot_price - 600.0) < 0.01

    def test_fallback_to_previous_price(self):
        """Test: Uses previous year's price if no new contracts."""
        contracts = []  # No new contracts
        previous_price = 650.0

        spot_price = calculate_state_spot_price(
            "PUNJAB",
            contracts,
            previous_spot_price=previous_price
        )

        assert spot_price == previous_price

    def test_fallback_to_aggregator_estimate(self):
        """Test: Estimates from aggregator if no contracts or previous price."""
        contracts = []
        mock_aggregator = Mock()
        mock_aggregator.feedstock_price = 450.0

        spot_price = calculate_state_spot_price(
            "PUNJAB",
            contracts,
            aggregator=mock_aggregator
        )

        # Should be ~30% above feedstock price
        expected = 450.0 * 1.3  # = 585.0
        assert abs(spot_price - expected) < 0.01

    def test_fallback_to_default(self):
        """Test: Uses default price as final fallback."""
        contracts = []
        default = 700.0

        spot_price = calculate_state_spot_price(
            "PUNJAB",
            contracts,
            default_price=default
        )

        assert spot_price == default


class TestGetContractPriceForYear:
    """Test contract price escalation helper function."""

    def test_year_zero_equals_initial(self):
        """Test: Price at year 0 equals initial price."""
        initial = 600.0
        price = get_contract_price_for_year(initial, years_elapsed=0)
        assert abs(price - initial) < 0.01

    def test_escalation_formula(self):
        """Test: Verify p_0 × (1 + r)^t formula."""
        initial = 600.0
        rate = 0.03
        years = 5

        price = get_contract_price_for_year(initial, years, rate)
        expected = initial * ((1 + rate) ** years)  # = 695.46

        assert abs(price - expected) < 0.01

    def test_escalation_year_10(self):
        """Test: Price after 10 years with 3% escalation."""
        initial = 600.0
        price = get_contract_price_for_year(initial, years_elapsed=10)
        expected = 600.0 * (1.03 ** 10)  # = 805.88

        assert abs(price - expected) < 0.01

    def test_escalation_year_20(self):
        """Test: Price after 20 years (end of typical contract)."""
        initial = 600.0
        price = get_contract_price_for_year(initial, years_elapsed=20)
        expected = 600.0 * (1.03 ** 20)  # = 1083.85

        assert abs(price - expected) < 0.01

    def test_different_escalation_rates(self):
        """Test: Function works with different escalation rates."""
        initial = 600.0
        years = 5

        test_cases = [
            (0.02, 600.0 * (1.02 ** 5)),  # 2% → 662.49
            (0.03, 600.0 * (1.03 ** 5)),  # 3% → 695.46
            (0.05, 600.0 * (1.05 ** 5)),  # 5% → 765.77
        ]

        for rate, expected in test_cases:
            price = get_contract_price_for_year(initial, years, rate)
            assert abs(price - expected) < 0.01

    def test_negative_years_raises_error(self):
        """Test: Negative years_elapsed raises ValueError."""
        with pytest.raises(ValueError, match="years_elapsed must be non-negative"):
            get_contract_price_for_year(600.0, years_elapsed=-1)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])

# CLAUDE END - Unit tests for contract pricing functions in utils.py
