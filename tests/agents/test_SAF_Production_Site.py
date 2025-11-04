import pytest

from unittest.mock import Mock

from src.Agents.SAF_Production_Site import SAFProductionSite

from src.Agents.Feedstock_Aggregator import FeedstockAggregator

from mesa import Model

 

class DummyModel(Model):

    """Minimal Mesa model with config and schedule time."""

 

    def __init__(self, time=0):

        self.config = {"saf_plant_construction_time": 2}

        self.schedule = Mock()

        self.schedule.time = time

 

@pytest.fixture

def dummy_aggregator():

    """Provides a mock FeedstockAggregator with fixed values."""

    aggregator = Mock(spec=FeedstockAggregator)

    aggregator.feedstock_price = 100.0

    aggregator.annual_load_factor = 0.8

    return aggregator

 

@pytest.fixture

def dummy_model():

    """Provides a dummy model with construction time and tick."""

    return DummyModel(time=0)

 

@pytest.fixture

def saf_site(dummy_model, dummy_aggregator):

    """Creates a SAFProductionSite with valid parameters."""

    return SAFProductionSite(

        unique_id=SAFProductionSite.generate_site_id("TX"),

        model=dummy_model,

        state_id="TX",

        investor_id="INV001",

        max_capacity=1000.0,

        design_load_factor=0.9,

        opex=50.0,

        aggregator=dummy_aggregator,

        capex_schedule=[100, 200, 300],

    )

 

def test_initialisation(saf_site):

    """

    Tests correct initialization of SAFProductionSite attributes.

    """

    assert saf_site.state_id == "TX"

    assert saf_site.investor_id == "INV001"

    assert saf_site.max_capacity == 1000.0

    assert saf_site.design_load_factor == 0.9

    assert saf_site.opex == 50.0

    assert saf_site.feedstock_price == 100.0

    assert saf_site.annual_load_factor == 0.8

    assert saf_site.site_id.startswith("site_TX_")

 

@pytest.mark.parametrize(

    "max_capacity, design_load_factor, opex, expected_exception",

    [

        (-10, 0.9, 50.0, ValueError),

        (1000, 1.5, 50.0, ValueError),

        (1000, 0.9, -5.0, ValueError),

    ],

)

def test_invalid_parameters(

    dummy_model,

    dummy_aggregator,

    max_capacity,

    design_load_factor,

    opex,

    expected_exception,

):

    with pytest.raises(expected_exception):

        SAFProductionSite(

            unique_id=SAFProductionSite.generate_site_id("TX"),

            model=dummy_model,

            state_id="TX",

            investor_id="INV001",

            max_capacity=max_capacity,

            design_load_factor=design_load_factor,

            opex=opex,

            aggregator=dummy_aggregator,

            capex_schedule=[],

        )

 

def test_calculate_srmc(saf_site):

    """

    Tests SRMC calculation as feedstock_price + opex.

    """

    assert saf_site.calculate_srmc() == 150.0

 

def test_calculate_production_output(saf_site):

    """

    Tests production output calculation using capacity, DLF, and ALF.

    """

    expected_output = 1000.0 * 0.9 * 0.8

    assert saf_site.calculate_production_output() == pytest.approx(expected_output)

 

def test_produce_before_construction(dummy_model, dummy_aggregator):

    """

    Tests that production is zero before construction is complete.

    """

    site = SAFProductionSite(

        unique_id=SAFProductionSite.generate_site_id("TX"),

        model=dummy_model,

        state_id="TX",

        investor_id="INV001",

        max_capacity=1000,

        design_load_factor=0.9,

        opex=50.0,

        aggregator=dummy_aggregator,

        capex_schedule=[],

    )

    dummy_model.schedule.time = 1  # Still under construction

    site.produce()

    assert site.production_output == 0.0

    assert site.srmc == 0.0

 

def test_produce_after_construction(dummy_model, dummy_aggregator):

    """

    Tests that production and SRMC are updated after construction.

    """

    dummy_model.schedule.time = 5  # Construction complete

    site = SAFProductionSite(

        unique_id=SAFProductionSite.generate_site_id("TX"),

        model=dummy_model,

        state_id="TX",

        investor_id="INV001",

        max_capacity=1000,

        design_load_factor=0.9,

        opex=50.0,

        aggregator=dummy_aggregator,

        capex_schedule=[],

    )

    site.tick_built = 0  # Ensure construction started at tick 0

    site.produce()

    assert site.production_output == pytest.approx(1000 * 0.9 * 0.8)

    assert site.srmc == 150.0

 

def test_site_id_uniqueness(dummy_model, dummy_aggregator):

    """

    Tests that each site gets a unique site_id.

    """

    site1 = SAFProductionSite(

        unique_id=SAFProductionSite.generate_site_id("TX"),

        model=dummy_model,

        state_id="TX",

        investor_id="INV001",

        max_capacity=1000,

        design_load_factor=0.9,

        opex=50.0,

        aggregator=dummy_aggregator,

        capex_schedule=[],

    )

    site2 = SAFProductionSite(

        unique_id=SAFProductionSite.generate_site_id("TX"),

        model=dummy_model,

        state_id="TX",

        investor_id="INV001",

        max_capacity=1000,

        design_load_factor=0.9,

        opex=50.0,

        aggregator=dummy_aggregator,

        capex_schedule=[],

    )

    assert site1.site_id != site2.site_id

 

