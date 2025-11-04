from typing import List, Dict
from src.Agents.SAF_Production_Site import SAFProductionSite
import logging
import math
 
logger = logging.getLogger(__name__)
 
def calculate_consumer_price(
    production_sites: List[Dict[str, float]],
    demand_this_tick: float,
    atf_plus_price: float,
) -> float:
    """
    Calculate the SAF consumer (clearing) price using a merit-order approach.
 
    Logic:
      - Sort operational supply sites by SRMC (ascending).
      - Accumulate supply until (>=) demand; the SRMC of the marginal unit sets the price.
      - If no supply is available (all filtered out) => price capped at atf_plus_price.
      - If demand <= 0 => return lowest SRMC (or cap).
 
    The function returns the marginal price (first value) and details about the marginal block (second value).
 
    Details include:
      - supply_at_marginal: Total supply from sites at the marginal price.
      - needed_from_marginal: Amount needed from the marginal price block to meet demand.
      - percentage_sold: Fraction of the marginal block that was actually sold.
 
    We note that all site with a higher SRMC than the ATF+ price are excluded from the merit order, as well as sites with zero production output.
 
    Parameters:
        production_sites: List of dicts each containing keys:
                             'srmc' (float) and 'production_output' (float).
        demand_this_tick: Demand for SAF (float). Values <= 0 treated as zero-demand edge case.
        atf_plus_price: Policy / cap price (float) acting as upper bound.
    Returns:
        (clearing_price (float), marginal_block_percentage (float in [0,1])).
    """
 
    merit_order = sorted(
        [
            (site["srmc"], site["production_output"])
            for site in production_sites
            if site["production_output"] > 0 and site["srmc"] <= atf_plus_price
        ],
        key=lambda x: x[0],
    )
    #no supply, hence ATF price is sold
    if not merit_order:
        details = {
            "supply_at_marginal": 0,
            "needed_from_marginal": 0,
            "percentage_sold": 0,
        }  
        return atf_plus_price, details
    #no demand, hence cheapest SRMC is considered marginal
    if demand_this_tick <= 0:
        details = {
            "supply_at_marginal": 0,
            "needed_from_marginal": 0,
            "percentage_sold": 1,
        }
        return merit_order[0][0], details
   
    total_supply = sum(output for srmc, output in merit_order)
 
    if total_supply < demand_this_tick:
        details = {
            "supply_at_marginal": 0,
            "needed_from_marginal": 0,
            "percentage_sold": 1,
        }
        return atf_plus_price, details
 
    #find marginal price where cumulative supply first meets demand
    cumulative_supply = 0.0
    consumer_price = atf_plus_price
 
    for srmc, output in merit_order:
        cumulative_supply += output
        if cumulative_supply >= demand_this_tick:
            consumer_price = srmc
            break
       
    supply_before_marginal = sum(output for srmc, output in merit_order if srmc < consumer_price)
    supply_at_marginal = sum(output for srmc, output in merit_order if math.isclose(srmc, consumer_price))
 
    needed_from_marginal=max(0, min((demand_this_tick - supply_before_marginal),supply_at_marginal))
 
    percentage_sold = min(1, needed_from_marginal / supply_at_marginal if supply_at_marginal > 0 else 0.0)
 
    details = {
        "supply_at_marginal": supply_at_marginal,
        "needed_from_marginal": needed_from_marginal,
        "percentage_sold": percentage_sold,
    }
 
    return consumer_price, details
 
def forecast_consumer_prices(
    model,
    production_sites: List[SAFProductionSite],
    demand_forecast: List[float],
    investment_horizon: int,
    atf_plus_price: float,
    current_tick: int,
) -> List[float]:
    """
    Forecast consumer prices over a forward horizon using projected operational capacity.
 
    Process per forecast year (t):
        - Determine which sites are (or will be) operational by year t.
        - Compute available production (design or realised production depending on prediction flag).
        - Run merit-order clearing with that year's demand to obtain price.
 
    Assumptions:
        - Demand beyond provided forecast list repeats last value.
        - Uses design load factor only (prediction mode) for forward realism.
 
    Parameters:
        production_sites: List of SAFProductionSite objects (operational and under construction).
        demand_forecast: List of annual demand values (float).
        investment_horizon: Number of future years to forecast (int >= 0).
        atf_plus_price: Upper price cap threshold (float).
        current_tick: Current simulation tick (int).
    Returns:
        List[float] of forecast prices length == years.
    """
    if investment_horizon < 0:
        raise ValueError("investment_horizon must be non-negative.")
 
    forecast = []
    model_start_year = model.config["start_year"]
    current_year = int(model_start_year + current_tick)
    end_year = int(current_year + int(investment_horizon))
 
    for year in range(current_year + 1, end_year + 1):
 
        tick_for_year = year - model_start_year
 
        operational_sites = find_operational_sites(
            production_sites, tick_for_year, prediction=True
        )
 
        demand = (
            get_saf_demand_forecast(year, model.config, demand_forecast)
        )
 
        price, details = calculate_consumer_price(
            operational_sites,
            demand,
            atf_plus_price,
        )
        forecast.append([price, [details]])
 
    return forecast
 
def find_operational_sites(production_sites, year, prediction=False):
    """
    Filter and map operational sites for a given year.
 
    Logic:
      - Includes sites with operational_year <= year..
      - Production output:
          * prediction=True  -> max_capacity * design_load_factor
          * prediction=False -> max_capacity * design_load_factor * current annual load factor
 
    Parameters:
        production_sites: List of SAFProductionSite objects.
        year: Year (tick) to test operational status against.
        prediction: If True, ignores realised annual load factor variability.
    Returns:
        List of dicts: { site_id, srmc, production_output }.
    """
    operational_sites = []
    for site in production_sites:
        if site.operational_year <= year:
            if not any(s["site_id"] == site.site_id for s in operational_sites):
                operational_sites.append(
                    {
                        "site_id": site.site_id,
                        "srmc": site.calculate_srmc(),
                        "production_output": (
                            site.max_capacity * site.design_load_factor
                            if prediction
                            else site.max_capacity
                            * site.design_load_factor
                            * site.aggregator.annual_load_factor
                            * site.streamday_percentage
                        ),
                    }
                )
    return operational_sites  
 
def get_saf_demand_forecast(year, config, atf_demand_forecast):
    """
    Resolve SAF demand for a given calendar year from a forecast dictionary.
 
    Logic:
      - Use the value for 'year' if present; otherwise fall back to the last available year.
      - Convert total ATF demand to SAF demand via config['blending_mandate'].
 
    Parameters:
        year: Calendar year to fetch (int).
        config: Model configuration (expects 'blending_mandate': float).
        atf_demand_forecast: Dict[int, float] mapping year -> total ATF demand.
 
    Returns:
        SAF demand for the requested year (float).
    """
    if atf_demand_forecast is None:
        raise ValueError("atf_demand_forecast cannot be None")
 
    if year in atf_demand_forecast:
        tick_total_fuel_demand = atf_demand_forecast[year]
    else:
        last_year = max(atf_demand_forecast.keys())
        tick_total_fuel_demand = atf_demand_forecast[last_year]
 
    tick_saf_demand = tick_total_fuel_demand * config["blending_mandate"]
 
    return tick_saf_demand
 
def year_for_tick(start_year, tick: int) -> int:
    """
    Map a zero-based model tick to a calendar year.
 
    Parameters:
        start_year: Simulation start year (int).
        tick: Zero-based model tick (int).
 
    Returns:
        Calendar year (int).
    """
    return int(start_year) + int(tick)
 
def tick_for_year(start_year, year: int) -> int:
    """
    Map a calendar year to a zero-based model tick.
 
    Parameters:
        start_year: Simulation start year (int).
        year: Calendar year (int).
 
    Returns:
        Zero-based model tick (int).
    """
    return int(year) - int(start_year)
 
 
# CLAUDE START - Contract pricing functions for Phase 1 implementation
def calculate_initial_contract_price(site) -> float:
    """
    Calculate initial contract price based on site's SRMC.

    In Phase 1, contract price equals the full SRMC (Short-Run Marginal Cost)
    of the production site at the time of contract signing.

    SRMC includes:
    - Feedstock price
    - OPEX (operating expenses)
    - Transport cost
    - Profit margin

    This price will then be escalated annually (handled by FeedstockContract).

    Parameters:
        site: SAFProductionSite instance

    Returns:
        Initial contract price in USD/tonne (equals site.srmc)

    Example:
        >>> site = SAFProductionSite(...)  # site with srmc = 600.0
        >>> price = calculate_initial_contract_price(site)
        >>> print(price)
        600.0
    """
    return site.srmc


def calculate_state_spot_price(
    state_id: str,
    new_contracts_this_year: List,
    previous_spot_price: float = None,
    aggregator = None,
    default_price: float = 700.0
) -> float:
    """
    Calculate annual spot price for a specific state.

    The spot price is the average of all new contract prices signed in that
    state during the current year. This creates a state-specific market price
    that reflects local supply/demand conditions.

    Logic:
    1. Filter contracts for this state from new_contracts_this_year
    2. If contracts exist: return average of their initial prices
    3. If no contracts: use previous_spot_price (carry forward)
    4. If no previous price: calculate from aggregator SRMC
    5. If no aggregator: use default_price as fallback

    Parameters:
        state_id: State identifier (e.g., "PUNJAB", "MAHARASHTRA")
        new_contracts_this_year: List of FeedstockContract objects signed this year
        previous_spot_price: Spot price from previous year (fallback)
        aggregator: FeedstockAggregator for this state (for SRMC calculation)
        default_price: Final fallback price if all else fails

    Returns:
        Spot price for this state in USD/tonne

    Example:
        >>> contracts = [
        ...     FeedstockContract(..., aggregator_id="PUNJAB", initial_contract_price=600.0),
        ...     FeedstockContract(..., aggregator_id="PUNJAB", initial_contract_price=620.0),
        ...     FeedstockContract(..., aggregator_id="MAHARASHTRA", initial_contract_price=550.0)
        ... ]
        >>> price = calculate_state_spot_price("PUNJAB", contracts)
        >>> print(price)
        610.0  # Average of 600 and 620
    """
    # Filter contracts for this specific state
    state_contracts = [
        c for c in new_contracts_this_year
        if c.aggregator_id == state_id
    ]

    # Case 1: New contracts exist - use their average
    if state_contracts:
        prices = [c.initial_contract_price for c in state_contracts]
        return sum(prices) / len(prices)

    # Case 2: No new contracts - use previous year's price
    if previous_spot_price is not None:
        logger.info(
            f"No new contracts for {state_id}, using previous spot price: "
            f"${previous_spot_price:.2f}/tonne"
        )
        return previous_spot_price

    # Case 3: No previous price - calculate from aggregator
    if aggregator is not None:
        # Create a temporary site to get SRMC (includes all cost components)
        # We can't import SAFProductionSite here due to circular imports,
        # so we calculate SRMC components directly
        feedstock_price = aggregator.feedstock_price

        # These would come from config, but for fallback we use reasonable defaults
        # In actual usage, aggregator should always be provided
        estimated_srmc = feedstock_price * 1.3  # ~30% markup for opex/transport/margin

        logger.info(
            f"No contracts or previous price for {state_id}, "
            f"using estimated SRMC: ${estimated_srmc:.2f}/tonne"
        )
        return estimated_srmc

    # Case 4: Complete fallback
    logger.warning(
        f"Cannot determine spot price for {state_id}, "
        f"using default: ${default_price:.2f}/tonne"
    )
    return default_price


def get_contract_price_for_year(
    initial_price: float,
    years_elapsed: int,
    escalation_rate: float = 0.03
) -> float:
    """
    Calculate escalated contract price for a specific year.

    This is a helper function that can be used independently of the
    FeedstockContract class for NPV calculations and forecasting.

    Formula: p(t) = p_0 × (1 + r)^t

    Parameters:
        initial_price: Contract price at signing (USD/tonne)
        years_elapsed: Years since contract start
        escalation_rate: Annual escalation rate (default: 0.03 = 3%)

    Returns:
        Escalated price in USD/tonne

    Example:
        >>> price_year_5 = get_contract_price_for_year(600.0, 5, 0.03)
        >>> print(f"${price_year_5:.2f}")
        $695.46
    """
    if years_elapsed < 0:
        raise ValueError(f"years_elapsed must be non-negative, got {years_elapsed}")

    return initial_price * ((1 + escalation_rate) ** years_elapsed)
# CLAUDE END - Contract pricing functions for Phase 1 implementation
