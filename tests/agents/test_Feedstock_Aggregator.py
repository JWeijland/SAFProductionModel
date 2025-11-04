import pytest

from unittest.mock import patch

from src.Agents.Feedstock_Aggregator import FeedstockAggregator

from mesa import Model

 

class DummyModel(Model):

    """Minimal Mesa model with config for multiplier bounds."""

 

    def __init__(self):

        self.config = {

            "feedstock_multiplier_min": 0.7,

            "feedstock_multiplier_max": 1.3,

        }

 

@pytest.fixture

def state_data():

    """Provides a dictionary of test states with varied feedstock prices and max supplies."""

    return {

        "TX": {"feedstock_price": 120.0, "max_supply": 100000},

        "NY": {"feedstock_price": 30.0, "max_supply": 0.0},

        "FL": {"feedstock_price": 0.0, "max_supply": 500.0},

        "ME": {"feedstock_price": 10.0, "max_supply": 1e-5},

        "AK": {"feedstock_price": 100.0, "max_supply": 1e12},

        "VT": {"feedstock_price": 50.0, "max_supply": 1.0},

        "IL": {"feedstock_price": 50.0, "max_supply": 1000.0},

    }

 

@pytest.fixture

def dummy_model():

    """Provides a dummy Mesa model with multiplier config."""

    return DummyModel()

 

@pytest.fixture

def aggregator(state_data, dummy_model):

    """Creates a FeedstockAggregator agent for TX with valid config."""

    return FeedstockAggregator(

        unique_id=1, model=dummy_model, state_id="TX", states_data=state_data

    )

 

@patch("src.Agents.Feedstock_Aggregator.random.uniform", return_value=1.0)

def test_initialisation(mock_random, state_data, dummy_model):

    """Tests correct initialisation and supply sampling with multiplier = 1.0."""

    agg = FeedstockAggregator(

        unique_id=1, model=dummy_model, state_id="TX", states_data=state_data

    )

    assert agg.state_id == "TX"

    assert agg.max_supply == 100000

    assert agg.feedstock_price == 120.0

    assert agg.current_supply == 100000

 

def test_sample_current_supply_range(aggregator):

    """Ensures sampled supply is within 70%–130% of max supply."""

    for _ in range(100):

        supply, load_factor = aggregator.sample_current_supply()

        assert 0.7 * aggregator.max_supply <= supply <= aggregator.max_supply

        assert 0.7 <= load_factor <= 1.0

 

def test_update_supply_changes_value(aggregator):

    """Checks that supply changes after calling update_supply()."""

    old_supply = aggregator.current_supply

    changed = False

    for _ in range(10):

        aggregator.update_supply()

        if aggregator.current_supply != old_supply:

            changed = True

            break

    assert changed, "Supply did not change after multiple updates."

 

@patch("src.Agents.Feedstock_Aggregator.random.uniform", return_value=0.7)

def test_sample_current_supply_min(mock_random, aggregator):

    """Tests sampling at minimum multiplier (0.7)."""

    supply, _ = aggregator.sample_current_supply()

    assert supply == pytest.approx(0.7 * aggregator.max_supply)

 

@patch("src.Agents.Feedstock_Aggregator.random.uniform", return_value=1.3)

def test_sample_current_supply_max(mock_random, aggregator):

    """Tests sampling at maximum multiplier (1.3), but supply is capped at max_supply."""

    supply, _ = aggregator.sample_current_supply()

    assert supply == aggregator.max_supply

 

def test_zero_max_supply(state_data, dummy_model):

    """Tests that zero max supply results in zero current supply."""

    agg = FeedstockAggregator(

        unique_id=2, model=dummy_model, state_id="NY", states_data=state_data

    )

    assert agg.current_supply == 0.0

 

def test_negative_max_supply(dummy_model):

    """Tests that negative max supply raises ValueError."""

    bad_data = {"FL": {"feedstock_price": 25.0, "max_supply": -100.0}}

    with pytest.raises(ValueError):

        FeedstockAggregator(

            unique_id=3, model=dummy_model, state_id="FL", states_data=bad_data

        )

 

def test_negative_feedstock_price(dummy_model):

    """Tests that negative feedstock price raises ValueError."""

    bad_data = {"WA": {"feedstock_price": -10.0, "max_supply": 500.0}}

    with pytest.raises(ValueError):

        FeedstockAggregator(

            unique_id=4, model=dummy_model, state_id="WA", states_data=bad_data

        )

 

@patch("src.Agents.Feedstock_Aggregator.random.uniform", return_value=1.3)

def test_extremely_high_max_supply(mock_random, state_data, dummy_model):

    """Tests that extremely high max supply respects multiplier cap."""

    agg = FeedstockAggregator(

        unique_id=5, model=dummy_model, state_id="AK", states_data=state_data

    )

    assert agg.current_supply == state_data["AK"]["max_supply"]

 

def test_non_string_state_id(state_data, dummy_model):

    """Tests that non-string state_id raises TypeError."""

    with pytest.raises(TypeError):

        FeedstockAggregator(

            unique_id=6, model=dummy_model, state_id=123, states_data=state_data

        )

 

def test_small_max_supply(state_data, dummy_model):

    """Tests that small max supply is handled correctly."""

    agg = FeedstockAggregator(

        unique_id=7, model=dummy_model, state_id="VT", states_data=state_data

    )

    assert 0.7 <= agg.current_supply <= 1.0

 

def test_zero_feedstock_price(state_data, dummy_model):

    """Tests that zero feedstock price is valid and handled correctly."""

    agg = FeedstockAggregator(

        unique_id=8, model=dummy_model, state_id="FL", states_data=state_data

    )

    assert agg.feedstock_price == 0.0

 

@patch("src.Agents.Feedstock_Aggregator.random.uniform", return_value=0.7)

def test_precision_low_supply(mock_random, state_data, dummy_model):

    """Tests precision handling for very small supply values."""

    agg = FeedstockAggregator(

        unique_id=9, model=dummy_model, state_id="ME", states_data=state_data

    )

    assert agg.current_supply == pytest.approx(0.7e-5)

 

@pytest.mark.parametrize("max_supply", ["1000", [1000], None])

def test_invalid_max_supply_type(max_supply, dummy_model):

    """Tests that invalid max_supply types raise TypeError."""

    bad_data = {"IL": {"feedstock_price": 50.0, "max_supply": max_supply}}

    with pytest.raises(TypeError):

        FeedstockAggregator(

            unique_id=10, model=dummy_model, state_id="IL", states_data=bad_data

        )

 

@pytest.mark.parametrize("feedstock_price", ["50", {"price": 50}, None])

def test_invalid_feedstock_price_type(feedstock_price, dummy_model):

    """Tests that invalid feedstock_price types raise TypeError."""

    bad_data = {"IL": {"feedstock_price": feedstock_price, "max_supply": 1000.0}}

    with pytest.raises(TypeError):

        FeedstockAggregator(

            unique_id=11, model=dummy_model, state_id="IL", states_data=bad_data

        )

 

