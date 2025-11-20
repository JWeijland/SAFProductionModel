import pytest

from unittest.mock import Mock

from collections import deque

from src.Agents.Investor import Investor

from src.Agents.Feedstock_Aggregator import FeedstockAggregator

from src.Agents.SAF_Production_Site import SAFProductionSite

import numpy as np

 

# --- Fixtures ---

 

@pytest.fixture

def mock_model():

    """Provides a mock model with required config and schedule attributes."""

    model = Mock()

    model.next_id.side_effect = range(1000, 2000)

    model.market_price = 200

    model.schedule.time = 2025

    model.production_sites = []

    model.schedule.add = Mock()

    model.config = {

        "DR_sample_min": 0.04,

        "DR_sample_max": 0.08,

        "Optimism_factor_sample_min": 0.8,

        "Optimism_factor_sample_max": 1.2,

        "capex_total_cost": 10,

        "saf_plant_construction_time": 3,

        "alpha": 1000,

        "Investment_horizon_length": 20,

        "max_plant_capacity": 100,

        "opex": 5,

        "beta": 0.01,

        "ideal_roace": 0.08,

        "DR_min": 0.01,

        "DR_max": 0.2,

        "max_change_dr": 0.005,

        "feedstock_multiplier_min": 0.7,

        "feedstock_multiplier_max": 1.3,

        "capex_annual_decrease": 0.01,

    }

    model.booleans = {"capex_decrease": False}

    model.consumer_price_forecast = [500] * 20

    model.states_available_feedstock = {"TX": 80, "CA": 60}

    model.aggregators = {}

    return model

 

@pytest.fixture

def sample_data():

    """Provides sample input data for investor and site evaluation."""

    return {

        "capex_schedule": [10, 5, 3] + [0] * 17,

        "consumer_price_forecast": [500] * 20,

        "states_data": {

            "TX": {"feedstock_price": 10, "max_supply": 100},

            "CA": {"feedstock_price": 15, "max_supply": 60},

        },

        "available_feedstock": {

            "TX": 80,

            "CA": 60,

        },

        "opex": 5,

    }

 

@pytest.fixture

def investor(mock_model, sample_data):

    """Creates an Investor instance with sample data."""

    return Investor(

        unique_id=Investor.generate_investor_id(),

        model=mock_model,

        current_tick=25,

        states_data=sample_data["states_data"],

        discount_rate=0.06,

        optimism_factor=1.0,

    )

 

@pytest.fixture

def aggregators(sample_data, mock_model):

    """Creates FeedstockAggregator instances for each state."""

    return {

        state: FeedstockAggregator(

            unique_id=1000 + i,

            model=mock_model,

            state_id=state,

            states_data=sample_data["states_data"],

        )

        for i, state in enumerate(sample_data["states_data"])

    }

 

# --- Core Investor Logic Tests ---

 

def test_investor_initialisation(investor):

    """Checks investor ID generation and default parameter bounds."""

    assert investor.investor_id.startswith("inv_")

    assert 0.04 <= investor.discount_rate <= 0.08

    assert 0.8 <= investor.optimism_factor <= 1.2

    assert investor.owned_assets == []

 

def test_site_id_generation():

    """Ensures investor ID generator produces valid prefix."""

    assert Investor.generate_investor_id().startswith("inv_")

 

def test_get_forecast_price(investor):

    """Tests forecast price retrieval with fallback to last value."""

    assert investor.get_forecast_price(5) == 500

    assert investor.get_forecast_price(25) == 500

 

def test_calculate_npv(investor):

    """Tests NPV calculation with sample production and cost values."""

    npv = investor.calculate_npv(100, 20, [10] * 20)

    assert abs(npv - 550441.52) < 5  # calculated manually - 5 is a tolerance

 

def test_calculate_ebit(investor):

    """Tests EBIT calculation logic with known inputs."""

    site = Mock()

    site.max_capacity = 100

    site.design_load_factor = 0.9

    site.srmc = 50

    ebit = investor.calculate_ebit(

        site, market_price=200, capex=10, annual_load_factor=0.8

    )

    assert ebit == 14030  # calculated manually

 

def test_evaluate_investment_structure(investor, sample_data, mock_model):

    """Tests structure of returned asset dictionary from evaluation."""

    aggregator = FeedstockAggregator(999, mock_model, "TX", sample_data["states_data"])

    site = SAFProductionSite(

        999,

        mock_model,

        "TX",

        investor.investor_id,

        investor.max_capacity,

        0.8,

        sample_data["opex"],

        aggregator,

        capex_schedule=sample_data["capex_schedule"],

    )

    asset = investor.evaluate_investment(site, 25)

    assert set(asset.keys()) >= {

        "site_id",

        "state_id",

        "npv",

        "capex_schedule",

        "tick_built",

        "ebit_history",

    }

 

# --- Investment Mechanism Tests ---

 

def test_investment_mechanism_success(investor, sample_data, aggregators):

    """Tests that the investor successfully invests in a site with valid feedstock."""

    investor.investment_mechanism(sample_data["available_feedstock"], aggregators, 25)

    assert len(investor.owned_assets) == 1

 

def test_investment_mechanism_no_feedstock(investor, aggregators, sample_data):

    """Ensures no investment is made when feedstock is unavailable in all states."""

    investor.investment_mechanism(

        {state: 0 for state in sample_data["states_data"]},

        aggregators,

        25,

    )

    assert len(investor.owned_assets) == 0

 

@pytest.mark.parametrize("alpha, expected", [(1e6, 0), (7, 1)])

def test_investment_mechanism_npv_threshold(

    investor, sample_data, aggregators, alpha, expected

):

    """Tests investment decision logic based on NPV threshold."""

    investor.alpha = alpha

    investor.investment_mechanism(sample_data["available_feedstock"], aggregators, 25)

    assert len(investor.owned_assets) == expected

 

# --- Financial Update & Discount Rate Tests ---

 

def test_annual_update_and_ebit_tracking(investor, sample_data, aggregators):

    """Checks that annual updates correctly track EBIT and capital investment."""

    investor.investment_mechanism(sample_data["available_feedstock"], aggregators, 25)

    for year_offset in range(6):

        investor.model.schedule.time = 25 + 1 + year_offset

        investor.model.market_price = 200 + year_offset * 5

        for _, site in investor.owned_assets:

            site.annual_load_factor = 0.9

        investor.annual_update(

            investor.model.market_price, investor.model.schedule.time

        )

 

    for asset, _ in investor.owned_assets:

        assert isinstance(asset["ebit_history"], deque)

        assert len(asset["ebit_history"]) <= asset["ebit_history"].maxlen

        assert len(set(asset["ebit_history"])) > 1

 

def test_adjust_discount_rate(investor, mock_model):

    """Tests updated discount rate adjustment logic using fixtures."""

    investor.total_capital_invested = 1000

    investor.roace_history = deque([0.05, 0.07], maxlen=3)

    investor.owned_assets = [({"ebit_history": [80, 90, 100]}, None)]

    expected_roace = sum([80, 90, 100]) / investor.total_capital_invested  # = 0.27

    avg_roace = (0.05 + 0.07 + expected_roace) / 3  # = ~0.13

    delta_dr = (avg_roace - mock_model.config["ideal_roace"]) * mock_model.config[

        "beta"

    ]  # = ~0.0005

    if abs(delta_dr) < mock_model.config["max_change_dr"]:

        expected_rate = investor.discount_rate - delta_dr

    else:

        expected_rate = (

            investor.discount_rate

            - np.sign(delta_dr) * mock_model.config["max_change_dr"]

        )

    expected_rate = max(

        mock_model.config["DR_min"], min(expected_rate, mock_model.config["DR_max"])

    )

    investor.adjust_discount_rate()

    assert abs(investor.roace_history[-1] - expected_roace) < 1e-4

    assert abs(investor.discount_rate - expected_rate) < 1e-4

 

def test_discount_rate_boundaries(investor):

    """Ensures discount rate stays within defined min/max bounds."""

    investor.discount_rate = investor.max_dr

    investor.total_capital_invested = 1000

    investor.owned_assets = [({"ebit_history": [100, 100, 100]}, None)]

    investor.adjust_discount_rate()

    assert investor.min_dr <= investor.discount_rate <= investor.max_dr

    investor.discount_rate = investor.min_dr

    investor.adjust_discount_rate()

    assert investor.min_dr <= investor.discount_rate <= investor.max_dr

 

def test_site_production_during_construction(investor, sample_data, mock_model):

    """Tests that SAFProductionSite does not produce during construction years."""

    aggregator = FeedstockAggregator(999, mock_model, "TX", sample_data["states_data"])

    site = SAFProductionSite(

        999,

        mock_model,

        "TX",

        investor.investor_id,

        investor.max_capacity,

        0.8,

        mock_model.config["opex"],

        aggregator,

        capex_schedule=sample_data["capex_schedule"],

    )

    site.tick_built = mock_model.schedule.time

    site.construction_years = 2

    mock_model.schedule.time = site.tick_built + 1

    site.produce()

    assert site.production_output == 0.0

    mock_model.schedule.time = site.tick_built + 3

    site.produce()

    assert site.production_output > 0.0

 

def test_multiple_investments_over_time(investor, sample_data, aggregators):

    """Tests that investor can make multiple investments across simulation years."""

    for year in range(3):

        investor.model.schedule.time = 25 + year

        investor.investment_mechanism(

            sample_data["available_feedstock"],

            aggregators,

            investor.model.schedule.time,

        )

    assert len(investor.owned_assets) >= 1

 

def test_get_dynamic_capex_decrease(investor):

    """Tests dynamic capex calculation with annual decrease."""

    base_cost = 1000

    year = 5

    decrease = 0.03

    expected = base_cost * (1 - decrease * year)

    result = investor.get_dynamic_capex(base_cost, year, decrease)

    assert result == expected

 

def test_get_dynamic_capex_negative_year(investor):

    """Tests that negative year raises ValueError in dynamic capex."""

    with pytest.raises(ValueError):

        investor.get_dynamic_capex(1000, -1, 0.03)

 

def test_repr_method(investor):

    """Tests the __repr__ output for Investor."""

    repr_str = repr(investor)

    assert "Investor(id=" in repr_str

    assert f"discount_rate={investor.discount_rate:.4f}" in repr_str

    assert f"owned_assets={len(investor.owned_assets)}" in repr_str

 

def test_evaluate_and_invest_methods(investor, mock_model, sample_data, aggregators):

    """Tests that evaluate and invest methods run without error and update state."""

    investor.model.states_available_feedstock = sample_data["available_feedstock"]

    investor.model.aggregators = aggregators

    investor.model.schedule.time = 25

    investor.model.market_price = 200

    investor.evaluate()

    investor.invest()

    assert isinstance(investor.discount_rate, float)

    assert isinstance(investor.owned_assets, list)

 

 

 