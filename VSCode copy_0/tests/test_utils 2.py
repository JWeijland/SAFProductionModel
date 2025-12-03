import pytest

from src.utils import (

    calculate_consumer_price,

    forecast_consumer_prices,

    find_operational_sites,

)

 

class MockSite:

    """

    Mock class for SAFProductionSite to test utility functions.

    """

 

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

 

    def calculate_srmc(self):

        return self.srmc

 

    def calculate_production_output(self):

        return self.production_output

 

def test_find_operational_sites_basic():

    """

    Should return only sites operational in the given year.

    """

    sites = [

        MockSite(1.0, 500, operational_year=0, site_id="A"),

        MockSite(1.5, 600, operational_year=2, site_id="B"),

    ]

    result = find_operational_sites(sites, year=1)

    assert any(site["site_id"] == "A" for site in result)

    assert all("srmc" in site and "production_output" in site for site in result)

 

def test_find_operational_sites_none_operational():

    """

    Should return empty list if no sites are operational.

    """

    sites = [

        MockSite(1.0, 500, operational_year=5, site_id="A"),

        MockSite(1.5, 600, operational_year=6, site_id="B"),

    ]

    result = find_operational_sites(sites, year=1)

    assert result == []

 

def test_find_operational_sites_empty():

    """

    Should return empty list if input is empty.

    """

    result = find_operational_sites([], year=1)

    assert result == []

 

def test_calculate_consumer_price_basic():

    """

    Should return the correct market clearing price for sufficient supply.

    """

    sites = [

        {"srmc": 1.0, "production_output": 500},

        {"srmc": 1.5, "production_output": 600},

    ]

    price = calculate_consumer_price(sites, demand_this_tick=1000, atf_plus_price=2.0)

    assert price == 1.5

 

def test_calculate_consumer_price_no_sites():

    """

    Should return price cap if no sites are available.

    """

    price = calculate_consumer_price([], demand_this_tick=1000, atf_plus_price=2.0)

    assert price == 2.0

 

def test_calculate_consumer_price_all_above_cap():

    """

    Should return price cap if all sites have SRMC above cap.

    """

    sites = [

        {"srmc": 5.0, "production_output": 500},

        {"srmc": 6.0, "production_output": 600},

    ]

    price = calculate_consumer_price(sites, demand_this_tick=1000, atf_plus_price=2.0)

    assert price == 2.0

 

def test_calculate_consumer_price_zero_demand():

    """

    Should return lowest SRMC if demand is zero and supply exists.

    """

    sites = [

        {"srmc": 1.0, "production_output": 500},

        {"srmc": 1.5, "production_output": 600},

    ]

    price = calculate_consumer_price(sites, demand_this_tick=0, atf_plus_price=2.0)

    assert price == 1.0

 

def test_forecast_consumer_prices_basic():

    """

    Should forecast correct number of years and return floats.

    """

    sites = [

        MockSite(1.0, 500, operational_year=0, site_id="A"),

        MockSite(1.5, 600, operational_year=0, site_id="B"),

    ]

    forecast = forecast_consumer_prices(

        sites, [800, 1000, 1200], 3, 2.0, current_tick=0

    )

    assert len(forecast) == 3

    assert all(isinstance(price, float) for price in forecast)

 

def test_forecast_consumer_prices_negative_years():

    """

    Should raise ValueError for negative years.

    """

    sites = [

        MockSite(1.0, 500, operational_year=0, site_id="A"),

    ]

    with pytest.raises(ValueError):

        forecast_consumer_prices(sites, [800, 1000, 1200], -1, 2.0, current_tick=0)

 

def test_forecast_consumer_prices_short_demand_forecast():

    """

    Should use last demand value if forecast is shorter than years.

    """

    sites = [

        MockSite(1.0, 500, operational_year=0, site_id="A"),

    ]

    forecast = forecast_consumer_prices(sites, [800], 3, 2.0, current_tick=0)

    assert len(forecast) == 3

    assert all(isinstance(price, float) for price in forecast)

 

def test_forecast_consumer_prices_empty_sites():

    """

    Should return price cap for all years if no sites are available.

    """

    forecast = forecast_consumer_prices([], [800, 1000, 1200], 3, 2.0, current_tick=0)

    assert len(forecast) == 3

    assert all(price == 2.0 for price in forecast)