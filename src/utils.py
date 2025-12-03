from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING
import logging
import math

if TYPE_CHECKING:
    from src.Agents.SAF_Production_Site import SAFProductionSite
 
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

    # NOTE: atf_plus_price parameter now receives the ESCALATED ATF+ price for the current year.
    # The base ATF+ price (€2000 in 2024) escalates at inflation rate (3%/year) to remain
    # economically consistent with fossil fuel price evolution.

    Parameters:
        production_sites: List of dicts each containing keys:
                             'srmc' (float) and 'production_output' (float).
        demand_this_tick: Demand for SAF (float). Values <= 0 treated as zero-demand edge case.
        atf_plus_price: Policy / cap price (float) acting as upper bound (already escalated for current year).
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
 
    inflation_rate = float(model.config.get("inflation_rate", 0.03))
    base_atf_plus_price = float(atf_plus_price)  # This is the base price passed in

    for year in range(current_year + 1, end_year + 1):

        tick_for_year = year - model_start_year

        operational_sites = find_operational_sites(
            production_sites, tick_for_year, prediction=True, model=model
        )

        demand = (
            get_saf_demand_forecast(year, model.config, demand_forecast)
        )

        years_elapsed_forecast = year - int(model_start_year)
        escalated_atf_plus_price_forecast = base_atf_plus_price * ((1 + inflation_rate) ** years_elapsed_forecast)

        price, details = calculate_consumer_price(
            operational_sites,
            demand,
            escalated_atf_plus_price_forecast,
        )
        forecast.append([price, [details]])
 
    return forecast
 
def find_operational_sites(production_sites, year, prediction=False, model=None):
    """
    Filter and map operational sites for a given year.

    Logic:
      - Includes sites with operational_year <= year (already operational)
      - ALSO includes sites under construction that will become operational within forecast window
      - Production output:
          * prediction=True  -> max_capacity * design_load_factor (potential production for forecasts)
          * prediction=False -> site.year_production_output (ACTUAL production for market clearing)

    # Merit order now uses MARKET-ESCALATED SRMC with technological improvement.
    #
    # Key distinction:
    # 1. Merit order SRMC = market feedstock price (escalates at 2%/year with tech improvement)
    # 2. Contract SRMC = contract feedstock price (escalates at 3%/year, CPI-indexed)
    # 3. Old contracts become MORE expensive than market over time (1%/year delta)
    #
    # Result: Plants with old, escalated contracts:
    #   - Appear competitive in merit order (based on current market SRMC ~2% escalation)
    #   - Set market price based on current market costs
    #   - BUT must produce at LOSS if contract SRMC (3% escalation) > market price
    #   - Contract obligation forces production even when unprofitable
    #
    # This creates realistic "stranded asset" dynamics where plants locked into
    # expensive long-term contracts lose competitiveness as technology improves.
    # See Investor.calculate_ebit() for contract obligation enforcement.

    Parameters:
        production_sites: List of SAFProductionSite objects.
        year: Year (tick) to test operational status against.
        prediction: If True, ignores realised annual load factor variability.
        model: Optional model reference for getting current year (for SRMC escalation)
    Returns:
        List of dicts: { site_id, srmc, production_output }.
    """
    operational_sites = []

    current_year = None
    if model is not None:
        # year parameter is a TICK (0, 1, 2, ...), convert to calendar year
        start_year = int(model.config["start_year"])
        current_year = start_year + int(year)

    # CRITICAL FIX for overinvestment: Investors must see plants being built!
    #
    # Problem before: Forecasts only counted operational sites, creating 4-year blindspot
    # during construction. This caused cascading overinvestment:
    #   Year N: Scarcity forecast → 40 investors build
    #   Year N+1-3: Forecasts ignore construction → MORE investors build (!!!)
    #   Year N+4: Plants operational → Massive oversupply → Too late
    #
    # Solution: Count sites that WILL BE operational within the forecast year.
    # This creates self-correcting feedback:
    #   Year N: Scarcity forecast → 40 investors build
    #   Year N+1: Forecasts SEE construction → No scarcity → Investment stops ✓
    #
    # Implementation: Include sites where operational_year <= forecast_year
    # (regardless of whether they're operational NOW at current tick)

    for site in production_sites:
        # OLD: if site.operational_year <= year
        # NEW: Same condition, but year is the FORECAST year, not current year
        # This naturally includes plants under construction that become operational by forecast year
        if site.operational_year <= year:
            if not any(s["site_id"] == site.site_id for s in operational_sites):
                # For non-prediction mode (market clearing), we want the TRUE marginal cost
                # For prediction mode (forecasts), we keep weighted average for consistency with contracts
                use_marginal = not prediction

                calculated_srmc = site.calculate_srmc(
                    current_year=current_year,
                    use_marginal_cost=use_marginal
                )

                operational_sites.append(
                    {
                        "site_id": site.site_id,
                        # Uses MARGINAL feedstock cost for market clearing (not weighted average)
                        # This ensures market price reflects true short-run marginal cost
                        "srmc": calculated_srmc,
                        # Market price is based on TRUE capacity (max physical production capability).
                        #
                        # Following copy_0 principle:
                        # - prediction=True:  max_capacity × design_load_factor (design capacity)
                        # - prediction=False: max_capacity × design_load_factor × annual_load_factor
                        #
                        # IMPORTANT: EXCLUDES streamday_percentage (copy_0 principle)
                        # - Streamday represents operational inefficiency (maintenance, breakdowns)
                        # - Market capacity = technical potential with available feedstock
                        # - NOT reduced by operational losses
                        #
                        # Why TRUE capacity (no spot_utilization, no streamday)?
                        # 1. Consistent with copy_0 baseline (proven correct)
                        # 2. Avoids circular dependency (spot_utilization depends on market price)
                        # 3. Market clearing shows full supply curve available to market
                        # 4. Streamday/spot_utilization applied AFTER price is determined
                        #
                        # Note: annual_load_factor represents feedstock availability (variable supply).
                        # This is the maximum feedstock availability before allocation priorities.
                        #
                        # Example:
                        # - True capacity = 2.5M (full capacity with feedstock, no operational losses)
                        # - Actual production = ~1.7M (with streamday ~0.69 and demand allocation)
                        # - Demand = 2.1M
                        # - Market price from 2.5M vs 2.1M → Marginal SRMC (~1600) ✓
                        "production_output": (
                            site.max_capacity * site.design_load_factor
                            if prediction
                            else site.max_capacity * site.design_load_factor *
                                 site.aggregator.annual_load_factor
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

    NEW MECHANISM (Tier-Based):
    Spot price = current tier marginal cost + spot premium (10%)
    This reflects the price for non-contracted feedstock at current market capacity.

    The spot price is now directly derived from the aggregator's tier position,
    making it simple, transparent, and tied to capacity utilization.

    Logic:
    1. If aggregator exists: use aggregator.get_spot_price() (tier-based)
    2. If no aggregator: use previous_spot_price (carry forward)
    3. If no previous price: use default_price as fallback

    Parameters:
        state_id: State identifier (e.g., "PUNJAB", "MAHARASHTRA")
        new_contracts_this_year: List of FeedstockContract objects signed this year (DEPRECATED - not used)
        previous_spot_price: Spot price from previous year (fallback only)
        aggregator: FeedstockAggregator for this state (REQUIRED for tier-based pricing)
        default_price: Final fallback price if all else fails

    Returns:
        Spot price for this state in USD/tonne

    Example:
        >>> aggregator = FeedstockAggregator(...)
        >>> aggregator.cumulative_allocated = 120000  # At Tier 2 boundary
        >>> price = calculate_state_spot_price("PUNJAB", [], aggregator=aggregator)
        >>> print(price)
        660.0  # Tier 2 (600) * 1.10 = 660
    """
    # Case 1: Aggregator exists - use tier-based spot price (PREFERRED)
    if aggregator is not None:
        spot_price = aggregator.get_spot_price()
        logger.info(
            f"{state_id} spot price: ${spot_price:.2f}/tonne "
            f"(tier-based at {aggregator.cumulative_allocated:.0f} ton/year allocated)"
        )
        return spot_price

    # Case 2: No aggregator - use previous year's price
    if previous_spot_price is not None:
        logger.warning(
            f"No aggregator for {state_id}, using previous spot price: "
            f"${previous_spot_price:.2f}/tonne"
        )
        return previous_spot_price

    # Case 3: Complete fallback
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
