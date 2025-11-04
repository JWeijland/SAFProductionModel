# CLAUDE START - Integration tests for Phase 1 contract implementation
"""
Integration Tests for Contract System

Simple tests to verify contract system components work together.
NOT a full model test - just contract infrastructure validation.

Run with: pytest tests/test_contract_integration.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.Agents.FeedstockContract import FeedstockContract
from src.Agents.Feedstock_Aggregator import FeedstockAggregator
from src.Agents.Investor import Investor
from src.Agents.SAF_Production_Site import SAFProductionSite
from src.utils import calculate_state_spot_price
from unittest.mock import Mock


class TestContractIntegration:
    """Simple integration tests for contract components."""

    def test_investor_creates_contract_for_plant(self):
        """Test: Investor can create a contract for a plant."""
        # Create mock model
        mock_model = Mock()
        mock_model.config = {
            "contract_duration": 20,
            "contract_escalation_rate": 0.03,
            "contract_percentage_min": 0.80,
            "contract_percentage_max": 0.90,
            # CLAUDE START - Phase 2 FIX: Add start_year for escalation calculation
            "start_year": 2024,  # Model started in 2024
            # CLAUDE END - Phase 2 FIX
        }

        # Create mock aggregator
        mock_aggregator = Mock()
        mock_aggregator.state_id = "PUNJAB"
        mock_aggregator.feedstock_price = 450.0

        # Create mock plant
        mock_plant = Mock()
        mock_plant.site_id = "site_001"
        mock_plant.state_id = "PUNJAB"
        mock_plant.srmc = 600.0
        mock_plant.max_capacity = 100000.0
        mock_plant.design_load_factor = 0.85

        # Create investor
        investor = Mock()
        investor.investor_id = "inv_01"
        investor.model = mock_model
        investor.contracts = []
        # Mock decide_contract_percentage to return a valid float
        investor.decide_contract_percentage = Mock(return_value=0.85)

        # Use the actual create_contract method
        from src.Agents.Investor import Investor as RealInvestor
        contract = RealInvestor.create_contract(
            investor,
            aggregator=mock_aggregator,
            plant=mock_plant,
            current_year=2025
        )

        # Verify contract was created
        assert contract is not None
        assert contract.investor_id == "inv_01"
        assert contract.aggregator_id == "PUNJAB"
        assert contract.plant_id == "site_001"
        # CLAUDE START - Phase 2 FIX: Contract price is escalated from base
        # 2025 is 1 year after 2024 start, so: $450 * 1.03^1 = $463.50
        expected_price = 450.0 * (1.03 ** 1)
        assert abs(contract.initial_contract_price - expected_price) < 0.01
        # CLAUDE END - Phase 2 FIX
        assert contract.start_year == 2025
        assert contract.end_year == 2045
        assert 0.80 <= contract.contract_percentage <= 0.90

        print(f"✓ Investor created contract with {contract.contract_percentage:.1%} coverage")
        print(f"  Initial contract price: ${contract.initial_contract_price:.2f} (escalated from base $450.00)")

    def test_aggregator_registers_and_tracks_contracts(self):
        """Test: Aggregator can register and track multiple contracts."""
        # Create mock model
        mock_model = Mock()
        mock_model.config = {"feedstock_multiplier_min": 0.8, "feedstock_multiplier_max": 1.2}

        # Create aggregator
        states_data = {"PUNJAB": {"max_supply": 1000000, "feedstock_price": 450.0}}
        aggregator = FeedstockAggregator(
            unique_id="agg_PUNJAB",
            model=mock_model,
            state_id="PUNJAB",
            states_data=states_data
        )

        # Create some contracts
        contract1 = FeedstockContract(
            contract_id="c1",
            investor_id="inv_01",
            aggregator_id="PUNJAB",
            plant_id="site_001",
            initial_contract_price=600.0,
            start_year=2025,
            end_year=2045,
            annual_capacity=85000.0,
            contract_percentage=0.85
        )

        contract2 = FeedstockContract(
            contract_id="c2",
            investor_id="inv_02",
            aggregator_id="PUNJAB",
            plant_id="site_002",
            initial_contract_price=620.0,
            start_year=2026,
            end_year=2046,
            annual_capacity=90000.0,
            contract_percentage=0.88
        )

        # Register contracts
        aggregator.register_contract(contract1)
        aggregator.register_contract(contract2)

        # Verify tracking
        assert len(aggregator.active_contracts) == 2

        # Check contracted capacity
        capacity_2025 = aggregator.get_contracted_capacity(2025)
        assert capacity_2025 == 85000.0 * 0.85  # Only contract1 active

        capacity_2026 = aggregator.get_contracted_capacity(2026)
        expected = 85000.0 * 0.85 + 90000.0 * 0.88  # Both active
        assert abs(capacity_2026 - expected) < 1.0

        print(f"✓ Aggregator tracks {len(aggregator.active_contracts)} contracts")
        print(f"  Contracted capacity 2025: {capacity_2025:.0f} tonnes/year")
        print(f"  Contracted capacity 2026: {capacity_2026:.0f} tonnes/year")

    def test_spot_price_calculated_from_contracts(self):
        """Test: Spot price averages new contract prices correctly."""
        # Create contracts from different investors
        contracts = [
            FeedstockContract(
                contract_id="c1",
                investor_id="inv_01",
                aggregator_id="PUNJAB",
                plant_id="site_001",
                initial_contract_price=600.0,
                start_year=2025,
                end_year=2045,
                annual_capacity=85000.0,
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
                annual_capacity=90000.0,
                contract_percentage=0.88
            ),
            FeedstockContract(
                contract_id="c3",
                investor_id="inv_03",
                aggregator_id="MAHARASHTRA",  # Different state!
                plant_id="site_003",
                initial_contract_price=550.0,
                start_year=2025,
                end_year=2045,
                annual_capacity=80000.0,
                contract_percentage=0.82
            ),
        ]

        # Calculate spot price for PUNJAB
        spot_price = calculate_state_spot_price(
            state_id="PUNJAB",
            new_contracts_this_year=contracts,
        )

        # Should average only PUNJAB contracts
        expected = (600.0 + 620.0) / 2
        assert abs(spot_price - expected) < 0.01

        print(f"✓ Spot price correctly calculated: ${spot_price:.2f}/tonne")
        print(f"  (Average of ${600:.2f} and ${620:.2f}, ignoring MAHARASHTRA)")

    def test_investor_blended_cost_calculation(self):
        """Test: Investor correctly calculates blended feedstock costs."""
        # Create mock model
        mock_model = Mock()
        mock_model.config = {
            "contract_duration": 20,
            "contract_escalation_rate": 0.03,
            "contract_percentage_min": 0.80,
            "contract_percentage_max": 0.90,
        }

        # Create mock plant
        mock_plant = Mock()
        mock_plant.site_id = "site_001"

        # Create investor with a contract
        investor = Mock()
        investor.model = mock_model
        investor.contracts = [
            FeedstockContract(
                contract_id="c1",
                investor_id="inv_01",
                aggregator_id="PUNJAB",
                plant_id="site_001",
                initial_contract_price=600.0,
                start_year=2025,
                end_year=2045,
                annual_capacity=85000.0,
                contract_percentage=0.85
            )
        ]

        # Use actual get_feedstock_cost method
        from src.Agents.Investor import Investor as RealInvestor
        blended_cost = RealInvestor.get_feedstock_cost(
            investor,
            plant=mock_plant,
            current_year=2030,  # 5 years later
            spot_price=720.0
        )

        # Calculate expected
        contract_price_year_5 = 600.0 * (1.03 ** 5)  # ~695.46
        expected = 0.85 * contract_price_year_5 + 0.15 * 720.0

        assert abs(blended_cost - expected) < 0.01

        print(f"✓ Blended cost calculated correctly: ${blended_cost:.2f}/tonne")
        print(f"  (85% contract @ ${contract_price_year_5:.2f} + 15% spot @ $720.00)")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "-s"])

# CLAUDE END - Integration tests for Phase 1 contract implementation
