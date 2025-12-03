import pytest

from src.Model import SAFMarketModel

from src.Agents.Investor import Investor

from src.Agents.SAF_Production_Site import SAFProductionSite

from src.Agents.Feedstock_Aggregator import FeedstockAggregator

from src.utils import calculate_consumer_price, forecast_consumer_prices

 

@pytest.fixture

def sample_model():

    """Creates a sample SAFMarketModel instance with predefined state and agent data."""

    states_data = {

        "TX": {"feedstock_price": 1.5, "max_supply": 1000},

        "CA": {"feedstock_price": 2.0, "max_supply": 1200},

    }

    constants = {

        "alpha": 0,

        "opex": 1.0,

        "initial_market_price": 5.0,

        "initial_num_investors": 2,

        "initial_num_SAF_sites": 2,

        "max_plant_capacity": 100,

        "feedstock_multiplier_min": 0.7,

        "feedstock_multiplier_max": 1.3,

        "saf_plant_construction_time": 4,

        "DR_sample_min": 0.04,

        "DR_sample_max": 0.08,

        "Optimism_factor_sample_min": 0.8,

        "Optimism_factor_sample_max": 1.2,

        "Investment_horizon_length": 20,

        "beta": 0.01,

        "ideal_roace": 0.50,

        "DR_min": 0.01,

        "DR_max": 0.2,

        "max_change_dr": 0.05,

        "capex_total_cost": 1000,

        "atf_plus_price": 25,

    }

    booleans = {"operational_initially": True, "capex_decrease": False}

    return SAFMarketModel(

        states_data=states_data,

        config=constants,

        saf_demand_forecast=[500] * 20,

        booleans=booleans,

    )

 

# --- Initialization Tests ---

def test_model_initialization(sample_model):

    """Checks correct initialization of investors, production sites, and aggregators."""

    assert len(sample_model.investors) == 2

    assert len(sample_model.production_sites) == 2

    assert set(sample_model.aggregators.keys()) == {"TX", "CA"}

 

def test_staged_activation_order(sample_model):

    """Verifies the correct order of model stages in the simulation schedule."""

    assert sample_model.schedule.stage_list == [

        "update_supply",

        "produce",

        "evaluate",

        "invest",

    ]

 

# --- Investor Logic ---

def test_investor_evaluation(sample_model):

    """Tests investor's ability to evaluate investment and calculate NPV."""

    investor = sample_model.investors[0]

    site = sample_model.production_sites[0]

    asset = investor.evaluate_investment(site, sample_model.schedule.time)

    assert asset["npv"] >= 0

 

def test_discount_rate_adjustment(sample_model):

    """Ensures discount rate adjusts based on ROACE history and investment."""

    investor = sample_model.investors[0]

    initial_rate = investor.discount_rate

    investor.total_capital_invested = 1000

    investor.roace_history.extend([0.1, 0.09, 0.08])

    investor.adjust_discount_rate()

    assert 0.01 <= investor.discount_rate <= 0.2

    assert investor.discount_rate != initial_rate

 

@pytest.mark.parametrize("market_price, expected_type", [(2.5, float)])

def test_calculate_ebit(sample_model, market_price, expected_type):

    """Validates EBIT calculation for a given market price and capex."""

    investor = sample_model.investors[0]

    site = sample_model.production_sites[0]

    capex = investor.capex_schedule[0]

    ebit = investor.calculate_ebit(site, market_price, capex, annual_load_factor=1.0)

    assert isinstance(ebit, expected_type)

 

def test_calculate_npv(sample_model):

    """Tests NPV calculation based on production output and SRMC."""

    investor = sample_model.investors[0]

    site = sample_model.production_sites[0]

    output = site.max_capacity * site.design_load_factor

    srmc = site.aggregator.feedstock_price + site.opex

    npv = investor.calculate_npv(output, srmc, investor.capex_schedule)

    assert npv >= 0

 

def test_zero_feedstock_investment(sample_model):

    """Ensures no investment occurs when feedstock availability is zero."""

    sample_model.states_available_feedstock = {

        state: 0 for state in sample_model.states_data

    }

    investor = sample_model.investors[0]

    initial_assets = len(investor.owned_assets)

    investor.investment_mechanism(

        states_available_feedstock=sample_model.states_available_feedstock,

        aggregators=sample_model.aggregators,

        current_tick=0,

    )

    assert len(investor.owned_assets) == initial_assets

 

# --- Model Behavior ---

def test_feedstock_update(sample_model):

    """Checks that feedstock availability remains non-negative after a model step."""

    sample_model.step()

    assert all(

        available >= 0 for available in sample_model.states_available_feedstock.values()

    )

 

def test_multiple_ticks_behavior(sample_model):

    """Simulates multiple ticks and verifies asset growth and feedstock availability."""

    initial_assets = sum(len(inv.owned_assets) for inv in sample_model.investors)

    for _ in range(3):

        sample_model.step()

    final_assets = sum(len(inv.owned_assets) for inv in sample_model.investors)

    assert sample_model.schedule.time == 3

    assert all(

        available >= 0 for available in sample_model.states_available_feedstock.values()

    )

    assert final_assets >= initial_assets

 

   

# --- Utility Functions ---

class MockSAFProductionSite:

    """Mock class for SAFProductionSite used in utility function tests."""

 

    def __init__(self, srmc, production_output, tick_built=0, construction_years=0):

        self.srmc = srmc

        self.production_output = production_output

        self.tick_built = tick_built

        self.construction_years = construction_years

 

    def calculate_srmc(self):

        return self.srmc

 

    def calculate_production_output(self):

        return self.production_output

 

class MockSite:

    def __init__(

        self,

        srmc,

        production_output,

        operational_year=0,

        site_id="mock",

        max_capacity=1000,

        design_load_factor=1.0,

        annual_load_factor=1.0,

    ):

        self.srmc = srmc

        self.production_output = production_output

        self.operational_year = operational_year

        self.site_id = site_id

        self.max_capacity = max_capacity

        self.design_load_factor = design_load_factor

        self.annual_load_factor = annual_load_factor

        self.tick_built = 0

        self.construction_years = 0

 

    def calculate_srmc(self):

        return self.srmc

 

    def calculate_production_output(self):

        return self.production_output

 

@pytest.mark.parametrize(

    "sites, demand, atf_plus_price, expected",

    [

        (

            [

                {"srmc": 1.0, "production_output": 500},

                {"srmc": 1.5, "production_output": 600},

            ],

            1000,

            2.0,

            1.5,

        ),

        (

            [

                {"srmc": 1.0, "production_output": 300},

                {"srmc": 1.5, "production_output": 400},

            ],

            1000,

            2.0,

            2.0,

        ),

    ],

)

def test_calculate_consumer_price(sites, demand, atf_plus_price, expected):

    """Tests consumer price calculation under sufficient and insufficient supply scenarios."""

    price = calculate_consumer_price(sites, demand, atf_plus_price)

    assert price == expected

 

def test_forecast_consumer_prices():

    """Validates consumer price forecasting over multiple years based on site output and demand."""

    sites = [

        MockSite(1.0, 500, operational_year=0),

        MockSite(1.5, 600, operational_year=0),

    ]

    forecast = forecast_consumer_prices(

        sites, [800, 1000, 1200], 3, 2.0, current_tick=0

    )

    assert len(forecast) == 3

    assert all(isinstance(price, float) for price in forecast)

 

def test_new_investor_added(sample_model):

    """Tests that a new investor is added to the model if they invest."""

    initial_count = len(sample_model.investors)

    sample_model.states_available_feedstock = {

        state: 100 for state in sample_model.states_data

    }

    sample_model.new_investor()

    assert len(sample_model.investors) >= initial_count

 

def test_new_investor_no_investment(sample_model):

    """Tests that a new investor is not added if they do not invest."""

    initial_count = len(sample_model.investors)

    sample_model.states_available_feedstock = {

        state: 0 for state in sample_model.states_data

    }

    sample_model.new_investor()

    assert len(sample_model.investors) == initial_count

 

def test_update_consumer_price(sample_model):

    """Tests that update_consumer_price sets a valid market price."""

    sample_model.update_consumer_price()

    assert isinstance(sample_model.market_price, (float, int))

    assert sample_model.market_price >= 0

 

def test_generate_price_forecast(sample_model):

    forecast = sample_model.generate_price_forecast()

    assert isinstance(forecast, list)

    assert all(isinstance(price, (float, int)) for price in forecast)

    assert len(forecast) == sample_model.config["Investment_horizon_length"]