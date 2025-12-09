# SAF Market Model - Complete Technical Documentation

**Document Version:** 2.0
**Date:** December 4, 2025
**Authors:** Technical Analysis of VSCode copy_0 (Baseline) and VSCode_copy_0_3 (New Model)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Baseline Model (VSCode copy_0)](#2-baseline-model-vscode-copy_0)
3. [New Model (VSCode_copy_0_3)](#3-new-model-vscode_copy_0_3)
4. [Comparative Analysis](#4-comparative-analysis)
5. [Configuration Parameters](#5-configuration-parameters)
6. [Data Structures](#6-data-structures)

---

## 1. Introduction

This document provides a comprehensive technical specification of two versions of the SAF (Sustainable Aviation Fuel) Market Model:
- **Baseline Model**: A simplified agent-based model with spot-market only feedstock procurement
- **New Model**: An enhanced model with long-term feedstock contracts and tiered pricing

Both models simulate the emergence and development of a SAF production market over 25+ years, modeling the interactions between investors, production facilities, and feedstock suppliers across multiple U.S. states.

### 1.1 Core Framework

Both models use the **Mesa** agent-based modeling framework in Python, implementing:
- **RandomActivation Scheduler**: Agents act in random order each tick (year)
- **DataCollector**: Records model and agent-level metrics
- **Agent Types**: FeedstockAggregator, SAFProductionSite, Investor

### 1.2 Simulation Time

- **Time Unit**: 1 tick = 1 year
- **Typical Runtime**: 100-200 ticks (years 2024-2224)
- **Start Year**: Configurable (default: 2024)

---

## 2. Baseline Model (VSCode copy_0)

### 2.1 Overview

The baseline model represents a **pure spot market** for feedstock with:
- Fixed feedstock prices per state
- No long-term contracts
- Stochastic feedstock availability
- Merit-order market clearing for SAF pricing

### 2.2 Agent Classes

#### 2.2.1 FeedstockAggregator

**Purpose**: Represents a U.S. state's feedstock supply characteristics.

**Attributes**:
```python
state_id: str                    # State identifier (e.g., "TAMIL_NADU")
max_supply: float                # Maximum theoretical feedstock (tonnes/year)
feedstock_price: float           # Fixed feedstock price (USD/tonne)
multiplier_min: float            # Minimum stochastic multiplier (default: 1.0)
multiplier_max: float            # Maximum stochastic multiplier (default: 1.0)
current_supply: float            # Sampled supply this tick
annual_load_factor: float        # Proportion of max supply available (0-1)
available_feedstock: float       # Feedstock remaining after site pledges
```

**Key Methods**:

1. **`sample_current_supply() -> (float, float)`**
   ```
   Purpose: Sample stochastic feedstock availability

   Formula:
     multiplier = random.uniform(multiplier_min, multiplier_max)
     current_supply = min(max_supply, max_supply * multiplier)
     annual_load_factor = min(1.0, multiplier)

   Returns: (current_supply, annual_load_factor)
   ```

2. **`update_supply() -> None`**
   ```
   Purpose: Resample supply each tick

   Logic:
     IF available_feedstock < 1:
       Resample using sample_current_supply()
     ELSE:
       current_supply = max_supply
       annual_load_factor = 1.0
   ```

**Behavior**: Passive agent - no produce(), evaluate(), or invest() actions.

---

#### 2.2.2 SAFProductionSite

**Purpose**: Represents a single SAF production facility.

**Attributes**:
```python
site_id: str                     # Unique site identifier
state_id: str                    # State where site operates
investor_id: str                 # Owning investor ID
max_capacity: float              # Maximum production capacity (tonnes SAF/year)
design_load_factor: float        # Fixed efficiency (0 < dlf ≤ 1)
opex: float                      # Operational expenditure (USD/tonne SAF)
aggregator: FeedstockAggregator  # Reference to state's aggregator
capex_schedule: List[float]      # CAPEX payment schedule (empty if operational)
transport_cost: float            # Feedstock transport cost (USD/tonne)
profit_margin: float             # Profit margin per tonne (USD/tonne)
construction_years: int          # Construction duration (years)
tick_built: int                  # Tick when construction started
operational_year: int            # Tick when site becomes operational
srmc: float                      # Short-Run Marginal Cost (USD/tonne)
streamday_percentage: float      # Operating days fraction (0.95-0.98)
year_production_output: float    # Annual production (tonnes/year)
```

**Key Methods**:

1. **`calculate_srmc() -> float`**
   ```
   Purpose: Compute cost to produce one tonne of SAF

   Formula:
     SRMC = feedstock_price + opex + transport_cost + profit_margin

   Where:
     feedstock_price: State's fixed spot price
     opex: Operating costs per tonne
     transport_cost: Feedstock delivery cost per tonne
     profit_margin: Desired margin per tonne

   Example:
     feedstock_price = $400/tonne
     opex = $285.71/tonne
     transport_cost = $600/tonne
     profit_margin = $250/tonne
     => SRMC = $1,535.71/tonne
   ```

2. **`calculate_production_output() -> float`**
   ```
   Purpose: Calculate annual SAF production

   Formula:
     Production = max_capacity
                  × design_load_factor
                  × annual_load_factor
                  × streamday_percentage

   Factors:
     - max_capacity: Plant's nameplate capacity
     - design_load_factor: Fixed at construction (based on feedstock availability)
     - annual_load_factor: Variable (from aggregator's supply sampling)
     - streamday_percentage: Random(0.95, 0.98) - plant uptime

   Example:
     max_capacity = 100,000 tonnes/year
     design_load_factor = 0.8 (80% of max)
     annual_load_factor = 0.9 (feedstock 90% available this year)
     streamday_percentage = 0.96 (96% uptime)
     => Production = 100,000 × 0.8 × 0.9 × 0.96 = 69,120 tonnes/year
   ```

3. **`produce() -> None`**
   ```
   Purpose: Stage-produce: Calculate production for current tick

   Logic:
     1. Resample streamday_percentage
     2. IF operational_year > current_tick:
          year_production_output = 0.0
        ELSE:
          year_production_output = calculate_production_output()
   ```

4. **`sample_streamday_percentage() -> float`**
   ```
   Purpose: Sample plant operating days

   Formula:
     streamday_percentage = random.uniform(streamday_min, streamday_max)

   Default: random.uniform(0.95, 0.98)
   Represents: 95-98% of days the plant operates (maintenance downtime)
   ```

**Behavior**:
- **update_supply()**: No action (passthrough)
- **produce()**: Calculates annual production if operational
- **evaluate()**: No action (passthrough)
- **invest()**: No action (passthrough)

---

#### 2.2.3 Investor

**Purpose**: Capital allocator that builds and operates production sites.

**Attributes**:
```python
investor_id: str                        # Unique investor identifier
discount_rate: float                    # NPV discount rate (6-12%)
optimism_factor: float                  # Revenue forecast multiplier (0.8-1.2)
owned_assets: List[(Dict, Site)]        # (asset_info_dict, site_object) pairs
current_tick: int                       # Current simulation tick
inv_ebit_history: Deque[float]          # Last 3 EBIT values (maxlen=3)
capex_schedule: List[float]             # Standard CAPEX payment schedule
min_NPV_threshold: float                # Minimum NPV to invest (default: 0.0)
investment_horizon: int                 # NPV evaluation horizon (25 years)
total_capital_invested: float           # Cumulative CAPEX spent
roace_history: Deque[float]             # Last 3 ROACE values (maxlen=3)
states_data: Dict[str, Dict]            # State feedstock data
consumer_price_forecast: List[float]    # Forward SAF price forecast
```

**Key Concepts**:

**EBIT (Earnings Before Interest and Tax)**:
```
Purpose: Measure annual operating profit from all owned sites

Formula:
  EBIT = Σ(site_revenue - site_costs) for all owned sites

Where for each site:
  site_revenue = production_output × market_price
  site_costs = production_output × (feedstock_price + opex + transport_cost)

Note: profit_margin is NOT deducted from EBIT (it's part of SRMC for market pricing)

Example (single site):
  production = 70,000 tonnes
  market_price = $1,610/tonne
  feedstock_cost = $400/tonne
  opex = $285.71/tonne
  transport = $600/tonne

  revenue = 70,000 × $1,610 = $112,700,000
  costs = 70,000 × ($400 + $285.71 + $600) = $89,999,700
  EBIT = $22,700,300
```

**ROACE (Return on Average Capital Employed)**:
```
Purpose: Measure return on invested capital

Formula:
  ROACE = EBIT / total_capital_invested

Where:
  EBIT: Earnings before interest and tax (from all sites)
  total_capital_invested: Cumulative CAPEX spent on all sites

Example:
  EBIT = $22,700,300
  total_capital_invested = $300,000,000
  ROACE = 22,700,300 / 300,000,000 = 0.0757 = 7.57%

Interpretation:
  - ROACE > 10%: Strong performance
  - ROACE 6-9%: Acceptable range
  - ROACE < 6%: Poor performance
```

**NPV (Net Present Value)**:
```
Purpose: Evaluate investment attractiveness

Formula:
  NPV = -CAPEX + Σ(CF_t / (1 + r)^t) for t = 1 to T

Where:
  CAPEX: Total upfront capital expenditure
  CF_t: Cash flow in year t = (price_t - costs_t) × production_t
  r: Discount rate (investor's required return)
  T: Investment horizon (25 years)

Components:
  price_t: Forecast SAF price in year t (from consumer_price_forecast)
  costs_t: feedstock_price + opex + transport_cost
  production_t: Expected annual production

Example:
  CAPEX = $300,000,000
  production = 70,000 tonnes/year (constant)
  price_0 = $1,610/tonne, growing 2%/year
  costs = $1,285.71/tonne (constant)
  discount_rate = 8%
  horizon = 25 years

  Year 0 CF = -$300,000,000
  Year 1 CF = 70,000 × ($1,610 - $1,285.71) / 1.08^1 = $21,013,426
  Year 2 CF = 70,000 × ($1,642.20 - $1,285.71) / 1.08^2 = $21,414,313
  ...
  Year 25 CF = 70,000 × ($2,638.58 - $1,285.71) / 1.08^25 = $13,851,269

  NPV = -$300M + Σ(discounted CFs) = $45,231,850

  Decision: NPV > 0 → INVEST
```

**Key Methods**:

1. **`__init__(...)`**
   ```
   Purpose: Initialize investor with sampled characteristics

   Sampling:
     discount_rate = random.uniform(DR_sample_min, DR_sample_max)
                   = random.uniform(0.06, 0.12)  # 6-12%

     optimism_factor = random.uniform(Optimism_factor_sample_min,
                                      Optimism_factor_sample_max)
                     = random.uniform(1.0, 1.0)  # No optimism in baseline

   CAPEX Schedule:
     total_cost = config["capex_total_cost"]  # $300,000,000
     construction_time = config["saf_plant_construction_time"]  # 4 years
     capex_schedule = [total_cost / construction_time] * construction_time
                    = [$75M, $75M, $75M, $75M]
   ```

2. **`update_supply() -> None`**
   ```
   Purpose: Update investor state (passthrough in baseline)
   ```

3. **`produce() -> None`**
   ```
   Purpose: No production role (passthrough)
   ```

4. **`evaluate() -> None`**
   ```
   Purpose: Assess performance and adjust discount rate

   Algorithm:
     1. Calculate EBIT from all owned sites:
        EBIT = Σ(revenue - costs) for all operational sites

     2. Calculate current ROACE:
        IF total_capital_invested > 0:
          ROACE = EBIT / total_capital_invested
        ELSE:
          ROACE = 0.0

     3. Update ROACE history (keep last 3 values)

     4. Adjust discount rate based on performance:
        IF len(roace_history) >= 3:
          avg_roace = mean(roace_history)

          IF avg_roace < ROACE_stability_min:  # < 6%
            # Poor performance → increase required return
            delta = DR_sensitivity_parameter × (ideal_roace - avg_roace)
            new_DR = discount_rate + delta

          ELIF avg_roace > ROACE_stability_max:  # > 9%
            # Strong performance → decrease required return
            delta = DR_sensitivity_parameter × (avg_roace - ideal_roace)
            new_DR = discount_rate - delta

          ELSE:  # 6% ≤ avg_roace ≤ 9%
            # Stable performance → move toward target
            new_DR = discount_rate + 0.5 × (DR_target - discount_rate)

          # Apply bounds
          discount_rate = max(DR_min, min(DR_max, new_DR))

     5. Log performance metrics

   Parameters:
     ideal_roace = 0.075 (7.5%)
     DR_sensitivity_parameter = 0.2
     ROACE_stability_min = 0.06 (6%)
     ROACE_stability_max = 0.09 (9%)
     DR_target = 0.075 (7.5%)
     DR_min = 0.04 (4%)
     DR_max = 0.20 (20%)
   ```

5. **`invest() -> None`**
   ```
   Purpose: Evaluate and execute new site investments

   Algorithm:
     1. Check if investor can afford investment:
        IF total_capital_invested > 0 AND recent poor performance:
          RETURN (don't invest)

     2. Find best state (lowest feedstock price):
        best_state = argmin(feedstock_price) over all states

     3. Check if feedstock available:
        available = model.states_available_feedstock[best_state]
        IF available < max_capacity × 0.5:
          RETURN (insufficient feedstock)

     4. Calculate design load factor:
        design_load_factor = min(1.0, available / max_capacity)

     5. Build CAPEX schedule:
        IF capex_decrease enabled:
          capex = get_dynamic_capex(...)  # Learning curve
        ELSE:
          capex = config["capex_total_cost"]  # $300M fixed

        capex_schedule = [capex / construction_time] * construction_time

     6. Evaluate NPV:
        NPV = calculate_NPV(
          state=best_state,
          design_load_factor=design_load_factor,
          discount_rate=self.discount_rate,
          price_forecast=consumer_price_forecast,
          horizon=investment_horizon
        )

     7. Investment decision:
        IF NPV > min_NPV_threshold:
          a. Create new SAFProductionSite
          b. Add site to owned_assets
          c. Update total_capital_invested += capex
          d. Register site with model
          e. Update available feedstock in state
          f. Log investment
        ELSE:
          LOG("NPV too low, not investing")
   ```

6. **`calculate_NPV(...) -> float`**
   ```
   Purpose: Calculate Net Present Value of potential investment

   Inputs:
     state_id: str
     design_load_factor: float
     discount_rate: float
     price_forecast: List[float]  # 25 years of forecast prices
     horizon: int  # 25 years

   Formula:
     NPV = -CAPEX + Σ(CF_t / (1 + r)^t) for t = 1 to T

   Cash Flow Calculation:
     production = max_capacity × design_load_factor × ALF × streamday

     For each year t:
       price_t = price_forecast[t] × optimism_factor
       costs_t = feedstock_price + opex + transport_cost
       CF_t = production × (price_t - costs_t)
       PV_t = CF_t / (1 + discount_rate)^t

     NPV = -CAPEX + Σ(PV_t)

   Returns: NPV (float, can be negative)
   ```

**Behavior**:
- **update_supply()**: No action
- **produce()**: No action
- **evaluate()**: Calculates EBIT/ROACE, adjusts discount rate
- **invest()**: Evaluates states, calculates NPV, builds new sites if NPV > 0

---

### 2.3 Model Orchestration

#### 2.3.1 Initialization

**Sequence**:
```
1. Create scheduler (RandomActivation)

2. Create FeedstockAggregators (one per state):
   FOR each state in states_data:
     Create FeedstockAggregator(state_id, max_supply, feedstock_price)
     Add to schedule

3. Create initial Investors:
   FOR i in range(initial_num_investors):  # default: 1
     Sample discount_rate ~ Uniform(6%, 12%)
     Sample optimism_factor ~ Uniform(1.0, 1.0)
     Create Investor(...)
     Add to schedule

4. Create initial SAFProductionSites:
   FOR i in range(initial_num_SAF_sites):  # default: 1
     a. Assign to investor (round-robin)
     b. Choose random state
     c. Calculate available feedstock
     d. Calculate design_load_factor = min(1.0, available / max_capacity)
     e. Generate CAPEX schedule (4 years × $75M)
     f. Create SAFProductionSite(...)
     g. IF operational_initially:
          site.operational_year = 0  # Start operational
          capex_schedule = []  # No construction delay
        ELSE:
          site.operational_year = tick_built + len(capex_schedule)
     h. Call site.produce() to initialize production
     i. Add to schedule
     j. Update investor.owned_assets
     k. Update available feedstock in state

5. Generate initial price forecast:
   consumer_price_forecast = generate_price_forecast()  # 25-year outlook

6. Distribute forecast to all investors:
   FOR each investor:
     investor.consumer_price_forecast = consumer_price_forecast

7. Collect initial snapshot (tick 0):
   datacollector.collect(model)
```

---

#### 2.3.2 Step Sequence

**Each tick follows this exact order**:

```
STEP(tick):

1. UPDATE AVAILABLE FEEDSTOCK
   Purpose: Calculate feedstock remaining after site commitments

   states_available_feedstock = {}
   FOR each state:
     states_available_feedstock[state] = aggregator.max_supply

   FOR each production site:
     pledged = site.max_capacity × site.design_load_factor
     states_available_feedstock[site.state_id] -= pledged
     states_available_feedstock[site.state_id] = max(0, ...)

   FOR each aggregator:
     aggregator.available_feedstock = states_available_feedstock[state]

2. UPDATE SUPPLY (all agents)
   Purpose: Resample stochastic feedstock availability

   FOR each agent in schedule (random order):
     agent.update_supply()

   Effect:
     - FeedstockAggregators: Resample current_supply and annual_load_factor
     - SAFProductionSites: No action
     - Investors: No action

3. PRODUCE (all agents)
   Purpose: Calculate SAF production for this year

   FOR each agent in schedule (random order):
     agent.produce()

   Effect:
     - FeedstockAggregators: No action
     - SAFProductionSites: Calculate year_production_output
     - Investors: No action

4. UPDATE MARKET PRICE
   Purpose: Clear market via merit-order pricing

   a. Get current demand:
      current_year = start_year + current_tick
      demand = get_saf_demand_forecast(current_year, atf_demand_forecast)

   b. Collect operational sites:
      operational_sites = [
        {"srmc": site.srmc, "production_output": site.year_production_output}
        for site in production_sites
        if site.operational_year <= current_tick
      ]

   c. Calculate market price:
      market_price, marginal_details = calculate_consumer_price(
        production_sites=operational_sites,
        demand_this_tick=demand,
        atf_plus_price=config["atf_plus_price"]  # $2000/tonne cap
      )

   Merit Order Logic:
     - Sort sites by SRMC (ascending)
     - Accumulate supply until >= demand
     - Price = SRMC of marginal site (last site needed to meet demand)
     - If supply < demand: price = atf_plus_price ($2000)
     - If no supply: price = atf_plus_price

   d. Store:
      model.market_price = market_price
      model.demand = demand
      model.marginal_details = marginal_details

5. UPDATE PRICE FORECAST
   Purpose: Generate forward-looking price expectations

   consumer_price_forecast = generate_price_forecast(
     production_sites=all_sites,
     demand_forecast=atf_demand_forecast,
     investment_horizon=25,
     current_tick=current_tick
   )

   Forecast Logic:
     FOR each future year t in [1, 25]:
       a. Determine which sites will be operational by year t
       b. Calculate expected production (using design load factor only)
       c. Get demand forecast for year t
       d. Run merit-order clearing
       e. forecast[t] = clearing price

   Distribution:
     FOR each investor:
       investor.consumer_price_forecast = consumer_price_forecast
       investor.current_tick = current_tick

6. EVALUATE (all agents)
   Purpose: Assess performance and update strategies

   FOR each agent in schedule (random order):
     agent.evaluate()

   Effect:
     - FeedstockAggregators: No action
     - SAFProductionSites: No action
     - Investors: Calculate EBIT/ROACE, adjust discount_rate

7. INVEST (all agents)
   Purpose: Execute new investments

   FOR each agent in schedule (random order):
     agent.invest()

   Effect:
     - FeedstockAggregators: No action
     - SAFProductionSites: No action
     - Investors: Evaluate NPV, build new sites if profitable

   Note: Sequential execution creates first-mover advantage
         (earlier investors in random order get first pick of feedstock)

8. NEW INVESTOR ENTRY
   Purpose: Introduce potential new market participant

   new_investor()

   Logic:
     a. Create new Investor with sampled characteristics
     b. Add to schedule
     c. Investor.invest() immediately evaluates market
     d. IF NPV > 0: new investor enters with site
        ELSE: new investor exits immediately (removed from schedule)

9. ADVANCE SCHEDULER
   Purpose: Increment simulation time

   schedule.step()

   Effect:
     - Increments schedule.time by 1
     - All agents' current_tick updated to schedule.time

10. COLLECT DATA
    Purpose: Record model and agent metrics

    FOR each agent:
      agent.current_tick = schedule.time

    datacollector.collect(model)

    Metrics Collected:
      Model-level:
        - Tick, Year
        - Consumer_Price (market_price)
        - Demand
        - Total_Capacity (sum of all site capacities)
        - Actual_Production (sum of all site outputs)
        - Num_Investors, Num_Production_Sites

      Agent-level:
        - Site: production_output, SRMC, state_id, investor_id
        - Investor: EBIT, ROACE, discount_rate, num_owned_assets
        - Aggregator: current_supply, annual_load_factor
```

---

### 2.4 Market Clearing Mechanism

**Merit-Order Pricing Algorithm**:

```python
def calculate_consumer_price(production_sites, demand, atf_plus_price):
    """
    Determine SAF market price via merit-order dispatch.

    Merit order = sorted list of (SRMC, production_output) pairs
    """

    # Step 1: Filter and sort
    merit_order = [
        (site.srmc, site.production_output)
        for site in production_sites
        if site.production_output > 0 and site.srmc <= atf_plus_price
    ]
    merit_order.sort(key=lambda x: x[0])  # Sort by SRMC ascending

    # Step 2: Edge cases
    if not merit_order:
        return atf_plus_price  # No supply → cap price

    if demand <= 0:
        return merit_order[0][0]  # No demand → lowest SRMC

    # Step 3: Check if supply meets demand
    total_supply = sum(output for _, output in merit_order)
    if total_supply < demand:
        return atf_plus_price  # Undersupply → cap price

    # Step 4: Find marginal unit
    cumulative_supply = 0.0
    for srmc, output in merit_order:
        cumulative_supply += output
        if cumulative_supply >= demand:
            return srmc  # This site sets the price

    # Fallback (should never reach)
    return atf_plus_price
```

**Example**:
```
Demand = 150,000 tonnes
Sites (sorted by SRMC):
  Site A: SRMC = $1,400/t, Output = 50,000t
  Site B: SRMC = $1,500/t, Output = 60,000t
  Site C: SRMC = $1,600/t, Output = 55,000t
  Site D: SRMC = $1,750/t, Output = 40,000t

Merit Order Dispatch:
  Cumulative after A: 50,000t (< 150,000) → keep going
  Cumulative after B: 110,000t (< 150,000) → keep going
  Cumulative after C: 165,000t (≥ 150,000) → STOP

Market Price = $1,600/tonne (Site C's SRMC)

Revenues:
  Site A: 50,000t × $1,600 = $80,000,000
  Site B: 60,000t × $1,600 = $96,000,000
  Site C: 40,000t × $1,600 = $64,000,000 (only 40k needed from C)
  Site D: 0t × $1,600 = $0 (not dispatched)

Profits (per tonne):
  Site A: $1,600 - $1,400 = $200/t profit
  Site B: $1,600 - $1,500 = $100/t profit
  Site C: $1,600 - $1,600 = $0/t profit (marginal producer breaks even)
  Site D: Would lose $150/t → doesn't produce
```

---

### 2.5 Price Forecasting

**Purpose**: Generate 25-year forward price outlook for investor NPV calculations.

**Algorithm**:
```python
def generate_price_forecast(model, production_sites, investment_horizon=25):
    """
    Project future SAF prices assuming current sites + under-construction sites.
    """
    forecast = []
    current_tick = model.schedule.time

    for future_tick in range(current_tick + 1, current_tick + investment_horizon + 1):
        # Step 1: Determine operational sites at future_tick
        operational_sites = [
            site for site in production_sites
            if site.operational_year <= future_tick
        ]

        # Step 2: Calculate expected production (PREDICTION MODE)
        # Uses design load factor only (no stochastic annual load factor)
        production_data = [
            {
                "srmc": site.srmc,
                "production_output": site.max_capacity
                                     × site.design_load_factor
                                     × 1.0  # Assume full feedstock availability
                                     × 0.965  # Average streamday percentage
            }
            for site in operational_sites
        ]

        # Step 3: Get demand forecast
        future_year = model.config["start_year"] + future_tick
        demand = get_saf_demand_forecast(future_year, model.atf_demand_forecast)

        # Step 4: Run merit-order clearing
        price, _ = calculate_consumer_price(
            production_sites=production_data,
            demand_this_tick=demand,
            atf_plus_price=model.config["atf_plus_price"]
        )

        forecast.append(price)

    return forecast
```

**Key Assumptions**:
1. **No new investments**: Only existing + under-construction sites included
2. **Optimistic production**: Uses design load factor without stochastic variation
3. **Demand growth**: Follows external ATF demand forecast (e.g., "Surge" scenario)
4. **No market dynamics**: Static competition (no new entrants considered)

**Impact on Investor Behavior**:
- High forecast prices → Higher NPVs → More investment
- Low forecast prices → Lower NPVs → Less investment
- Creates expectation-driven boom/bust cycles

---

### 2.6 Configuration Parameters

**Baseline Model Config** (config.csv):
```
Parameter                    Default Value   Unit                  Description
-----------------------------------------------------------------------------------------
# Economic Parameters
min_NPV_threshold           0.0             USD                   Minimum NPV to invest
opex                        285.71          USD/tonne             Operating expenditure
transport_cost              600.0           USD/tonne             Feedstock transport cost
profit_margin               250.0           USD/tonne             Profit margin per tonne
capex_total_cost            300,000,000     USD                   Total plant CAPEX
atf_plus_price              2000.0          USD/tonne             SAF price cap

# Capacity Parameters
max_capacity                100,000         tonnes/year           Plant nameplate capacity
streamday_min               0.95            fraction              Min plant uptime
streamday_max               0.98            fraction              Max plant uptime

# Investor Parameters
initial_num_investors       1               count                 Starting investors
DR_sample_min               0.06            fraction              Min discount rate
DR_sample_max               0.12            fraction              Max discount rate
DR_min                      0.04            fraction              DR lower bound
DR_max                      0.20            fraction              DR upper bound
DR_target                   0.075           fraction              Target DR
DR_sensitivity_parameter    0.2             -                     DR adjustment speed
ideal_roace                 0.075           fraction              Target ROACE (7.5%)
ROACE_stability_min         0.06            fraction              Lower stability band
ROACE_stability_max         0.09            fraction              Upper stability band
Investment_horizon_length   25              years                 NPV horizon
Optimism_factor_sample_min  1.0             multiplier            Min price optimism
Optimism_factor_sample_max  1.0             multiplier            Max price optimism

# Site Parameters
initial_num_SAF_sites       1               count                 Starting sites
saf_plant_construction_time 4               years                 Construction duration

# Feedstock Parameters
feedstock_multiplier_min    1.0             multiplier            Min supply multiplier
feedstock_multiplier_max    1.0             multiplier            Max supply multiplier

# Time Parameters
start_year                  2024            year                  Simulation start
blending_mandate            0.01            fraction              SAF mandate (not used)

# Cost Evolution (disabled in baseline)
capex_annual_decrease       0.01            fraction/year         CAPEX learning rate
inflation_rate              0.0             fraction/year         CPI inflation
tech_improvement_rate       0.0             fraction/year         Tech cost reduction
```

**Boolean Flags** (booleans.csv):
```
Flag                         Default  Description
----------------------------------------------------
operational_initially        True     Sites start operational (no construction delay)
capex_decrease               False    Enable CAPEX learning curve
enable_demand_allocation     True     Allocate demand proportionally when oversupply
```

---

### 2.7 Data Outputs

**Model-Level Metrics**:
- `Tick`: Current simulation tick
- `Year`: Calendar year (start_year + tick)
- `Consumer_Price`: Market clearing price (USD/tonne)
- `Market_Price`: Same as Consumer_Price
- `Demand`: SAF demand this year (tonnes)
- `Total_Capacity`: Sum of all site effective capacities
- `Actual_Production`: Sum of all site production outputs
- `Num_Investors`: Count of active investors
- `Num_Production_Sites`: Count of production sites

**Agent-Level Metrics** (SAFProductionSite):
- `site_id`, `state_id`, `investor_id`
- `year_production_output`: Annual SAF production (tonnes)
- `srmc`: Short-run marginal cost (USD/tonne)
- `max_capacity`: Nameplate capacity (tonnes/year)
- `design_load_factor`: Fixed efficiency (0-1)
- `operational_year`: Tick when site became operational

**Agent-Level Metrics** (Investor):
- `investor_id`
- `EBIT`: Earnings before interest and tax (USD)
- `ROACE`: Return on average capital employed (fraction)
- `discount_rate`: Current NPV discount rate (fraction)
- `num_owned_assets`: Count of owned sites
- `total_capital_invested`: Cumulative CAPEX spent (USD)

**Agent-Level Metrics** (FeedstockAggregator):
- `state_id`
- `current_supply`: Sampled feedstock availability (tonnes/year)
- `annual_load_factor`: Supply availability fraction (0-1)
- `available_feedstock`: Remaining after site pledges (tonnes/year)
- `feedstock_price`: Fixed price (USD/tonne)

---

## 3. New Model (VSCode_copy_0_3)

### 3.1 Overview

The new model introduces **realistic commodity market features**:
- Long-term feedstock contracts (20-year duration)
- Tiered pricing system (supply curves)
- Contract escalation (3% annually)
- Take-or-pay commitment dynamics
- First-mover advantages in feedstock allocation

### 3.2 New Data Structures

#### 3.2.1 FeedstockContract

**Purpose**: Represents a long-term supply agreement between an aggregator and a production site.

**Structure** (dataclass):
```python
@dataclass
class FeedstockContract:
    # Required fields
    contract_id: str                    # Unique contract ID
    investor_id: str                    # Owning investor
    aggregator_id: str                  # State ID (e.g., "TAMIL_NADU")
    plant_id: str                       # Production site ID
    initial_contract_price: float       # Tier price at signing (USD/tonne)
    start_year: int                     # Contract start year
    end_year: int                       # Contract end year (start + duration)
    annual_capacity: float              # Plant effective capacity (tonnes/year)
    contract_percentage: float          # Coverage (0.80-0.90 typically)

    # Optional fields (with defaults)
    escalation_rate: float = 0.03       # Annual price increase (3%)
    duration: int = 20                  # Contract length (years)
    status: str = "active"              # "active" or "expired"
```

**Key Properties**:

1. **`contracted_volume -> float`**
   ```
   Purpose: Annual feedstock guaranteed by contract

   Formula:
     contracted_volume = annual_capacity × contract_percentage

   Example:
     annual_capacity = 80,000 tonnes/year
     contract_percentage = 0.85 (85%)
     contracted_volume = 68,000 tonnes/year
   ```

2. **`spot_volume -> float`**
   ```
   Purpose: Annual feedstock purchased on spot market

   Formula:
     spot_volume = annual_capacity × (1 - contract_percentage)

   Example:
     annual_capacity = 80,000 tonnes/year
     contract_percentage = 0.85
     spot_volume = 12,000 tonnes/year (15%)
   ```

3. **`get_price_for_year(current_year) -> float`**
   ```
   Purpose: Get escalated contract price for specific year

   Formula:
     price(t) = initial_contract_price × (1 + escalation_rate)^t

   Where:
     t = current_year - start_year

   Example:
     initial_contract_price = $600/tonne
     escalation_rate = 0.03 (3%)
     start_year = 2024
     current_year = 2029
     t = 5 years

     price(2029) = $600 × (1.03)^5 = $695.46/tonne
   ```

4. **`is_active(current_year) -> bool`**
   ```
   Purpose: Check if contract is currently valid

   Logic:
     RETURN (start_year <= current_year <= end_year) AND (status == "active")
   ```

**Contract Lifecycle**:
```
Year 0: Investment → Create contract
        - Sign at current tier price (e.g., $600/t in tier 2)
        - Lock in 85% of capacity
        - Duration: 20 years

Year 1-19: Active contract
        - Pay escalated price: price(t) = $600 × (1.03)^t
        - Guaranteed feedstock supply (priority allocation)
        - Must pay for contracted volume even if not producing

Year 20: Contract expires
        - Automatic renewal at same tier (no queue movement)
        - New 20-year contract at current tier price
        - OR switch to 100% spot market

Post-expiry options:
        - Renew: Get new contract at same tier
        - Spot: Buy 100% at current market price (higher risk)
```

---

#### 3.2.2 Tiered Pricing System

**Purpose**: Model feedstock supply curves - price increases as demand grows.

**Tier Structure**:
```
Tier Definition:
  Each tier covers a fixed capacity width (default: 120,000 tonnes/year)
  Tier price = base_tier_cost + (tier_number - 1) × tier_cost_increment

Example Configuration:
  tier_capacity_size = 120,000 tonnes/year
  tier_1_cost = $400/tonne
  tier_cost_increment = $200/tonne

Tier Prices:
  Tier 1: $400/t   (0 - 120,000 tonnes/year)
  Tier 2: $600/t   (120,000 - 240,000 tonnes/year)
  Tier 3: $800/t   (240,000 - 360,000 tonnes/year)
  Tier 4: $1000/t  (360,000 - 480,000 tonnes/year)
  ...
```

**State-Specific Tiers**:
```
Each state has independent tier system based on max_supply:

State: TAMIL_NADU
  max_supply = 600,000 tonnes/year
  Number of tiers = ceil(600,000 / 120,000) = 5 tiers

Tiers:
  1: [0, 120,000] @ $400/t
  2: [120,000, 240,000] @ $600/t
  3: [240,000, 360,000] @ $800/t
  4: [360,000, 480,000] @ $1000/t
  5: [480,000, 600,000] @ $1200/t

Total capacity across tiers: 600,000 tonnes/year
```

**Allocation Logic**:
```
cumulative_allocated: Tracks total feedstock allocated across all contracts
                     Persists across years (no reset)

When new contract signed:
  1. Determine required capacity from contract
  2. Start at current cumulative_allocated position
  3. Fill from current tier upward
  4. Advance cumulative_allocated
  5. Return weighted average price

Example:
  Current cumulative_allocated = 200,000 tonnes/year
  New contract requests = 100,000 tonnes/year

  Allocation:
    Tier 2 remaining = 240,000 - 200,000 = 40,000 tonnes @ $600/t
    Tier 3 needed = 100,000 - 40,000 = 60,000 tonnes @ $800/t

  Weighted price:
    (40,000 × $600 + 60,000 × $800) / 100,000 = $720/tonne

  Update cumulative_allocated:
    200,000 → 300,000 tonnes/year
```

**Contract Renewal Logic**:
```
When contract expires and renews:
  1. Calculate current tier for contract's original allocation position
  2. Get tier price (no escalation - resets to base tier price)
  3. Create new contract at that tier price
  4. Do NOT advance cumulative_allocated (stay in same position)

Example:
  Original contract:
    Signed in year 2024 at cumulative_allocated = 150,000
    Tier 2 price = $600/tonne
    20-year contract @ $600 with 3% escalation

  Year 2043 (after escalation):
    Price = $600 × (1.03)^19 = $1,048.25/tonne

  Year 2044 (renewal):
    Check tier at position 150,000 → Still tier 2
    New contract price = $600/tonne (resets, no escalation carried over)
    20-year contract @ $600 with 3% escalation (fresh start)
    cumulative_allocated remains at original position (no queue movement)

Rationale:
  - Rewards early movers (locked into lower tiers)
  - No penalty for renewal (don't get pushed to back of queue)
  - Simulates "grandfathered" supply agreements
```

---

### 3.3 Modified Agent Behaviors

#### 3.3.1 FeedstockAggregator (Enhanced)

**New Attributes**:
```python
# Tier system
tier_capacity_size: float                    # Capacity per tier (120,000 t/yr)
tier_1_cost: float                           # Base tier price ($400/t)
tier_cost_increment: float                   # Increment per tier ($200/t)
tiers: List[Tuple[float, float, float]]      # (start, end, price) for each tier
aggregator_profit_margin: float              # Fixed margin ($50/t)

# Allocation tracking
cumulative_allocated: float                  # Total allocated across contracts
contracts: List[FeedstockContract]           # Active contracts list
total_contracted_demand: float               # Sum of contracted volumes
total_spot_demand: float                     # Sum of spot volumes
contracted_load_factor: float                # Contract fulfillment ratio
spot_load_factor: float                      # Spot fulfillment ratio
```

**New Methods**:

1. **`generate_tiers() -> List[Tuple]`**
   ```
   Purpose: Create tier structure for this state

   Algorithm:
     num_tiers = ceil(max_supply / tier_capacity_size)
     tiers = []

     FOR i in range(num_tiers):
       start = i × tier_capacity_size
       end = min((i + 1) × tier_capacity_size, max_supply)
       price = tier_1_cost + i × tier_cost_increment
       tiers.append((start, end, price))

     RETURN tiers

   Example (max_supply = 600,000):
     tier_capacity_size = 120,000
     num_tiers = 5

     Tier 0: (0, 120,000, $400)
     Tier 1: (120,000, 240,000, $600)
     Tier 2: (240,000, 360,000, $800)
     Tier 3: (360,000, 480,000, $1000)
     Tier 4: (480,000, 600,000, $1200)
   ```

2. **`allocate_contract(contract_volume) -> float`**
   ```
   Purpose: Allocate feedstock to new contract and return price

   Algorithm:
     start_position = cumulative_allocated
     remaining = contract_volume
     total_cost = 0.0

     FOR each tier:
       IF start_position >= tier.end:
         CONTINUE  # Already past this tier

       available_in_tier = tier.end - max(tier.start, start_position)
       allocated_in_tier = min(available_in_tier, remaining)

       total_cost += allocated_in_tier × tier.price
       remaining -= allocated_in_tier

       IF remaining == 0:
         BREAK

     weighted_price = total_cost / contract_volume
     cumulative_allocated += contract_volume

     RETURN weighted_price

   Example:
     cumulative_allocated = 180,000
     contract_volume = 150,000

     Tier 1 (120k-240k @ $600):
       available = 240,000 - 180,000 = 60,000
       allocated = min(60,000, 150,000) = 60,000
       cost = 60,000 × $600 = $36,000,000
       remaining = 150,000 - 60,000 = 90,000

     Tier 2 (240k-360k @ $800):
       available = 360,000 - 240,000 = 120,000
       allocated = min(120,000, 90,000) = 90,000
       cost = 90,000 × $800 = $72,000,000
       remaining = 90,000 - 90,000 = 0

     Total cost = $36M + $72M = $108M
     Weighted price = $108M / 150,000 = $720/tonne

     cumulative_allocated = 180,000 + 150,000 = 330,000
   ```

3. **`renew_contract_at_same_tier(existing_contract, current_year) -> float`**
   ```
   Purpose: Renew contract at original tier without moving in queue

   Algorithm:
     # Find original allocation position
     position = calculate_contract_position(existing_contract)

     # Determine which tier this position is in
     tier = find_tier_for_position(position)

     # Return base tier price (no escalation)
     RETURN tier.price

   Example:
     existing_contract:
       allocated at cumulative = 150,000
       falls in tier 1 (120k-240k @ $600)

     renewal_price = $600/tonne (resets to base tier 1 price)

     Note: cumulative_allocated does NOT advance
           (contract stays in same queue position)
   ```

4. **`register_contract(contract) -> None`**
   ```
   Purpose: Add contract to aggregator's tracking

   Algorithm:
     contracts.append(contract)
     LOG contract registration
   ```

5. **`update_supply() -> None`**
   ```
   Purpose: Calculate load factors for contracted vs spot feedstock

   Algorithm:
     # Sample current supply (stochastic)
     current_supply, _ = sample_current_supply()

     # Calculate load factors
     IF total_contracted_demand > 0:
       contracted_load_factor = min(1.0,
                                    current_supply / total_contracted_demand)
     ELSE:
       contracted_load_factor = 1.0

     IF total_spot_demand > 0:
       remaining_supply = max(0, current_supply - total_contracted_demand)
       spot_load_factor = min(1.0,
                              remaining_supply / total_spot_demand)
     ELSE:
       spot_load_factor = 1.0

     # Set annual load factor (global fallback)
     annual_load_factor = contracted_load_factor

   Interpretation:
     contracted_load_factor: Fraction of contracted volume available
       - 1.0 = Full supply, all contracts honored
       - 0.8 = 80% supply, contracts get 80% of promised feedstock

     spot_load_factor: Fraction of spot demand fulfilled
       - Calculated AFTER contracted demand satisfied
       - Often lower than contracted_load_factor (contracts prioritized)
   ```

6. **`get_marginal_feedstock_price(current_position) -> float`**
   ```
   Purpose: Get spot price for next tonne at current allocation level

   Algorithm:
     FOR each tier:
       IF tier.start <= current_position < tier.end:
         RETURN tier.price

     # If beyond all tiers, return highest tier price
     RETURN tiers[-1].price

   Usage: Used for SRMC calculations in merit order
   ```

**Key Behavior Changes**:
- **Priority Allocation**: Contracted demand satisfied first, spot gets remainder
- **Tier-Based Pricing**: Each contract pays tier price (not state average)
- **Queue Persistence**: Early contracts maintain tier positions indefinitely

---

#### 3.3.2 SAFProductionSite (Enhanced)

**New Attributes**:
```python
active_contract: Optional[FeedstockContract]  # Current feedstock contract
allocated_demand: float                       # Demand assigned (when oversupply)
```

**Modified Methods**:

1. **`calculate_srmc(use_marginal_cost=False) -> float`**
   ```
   Purpose: Compute SRMC using appropriate feedstock cost

   TWO MODES:

   A) Market Clearing Mode (use_marginal_cost=False):
      Used for: Investor NPV calculations, revenue/cost tracking

      Feedstock Cost Calculation:
        IF has active contract:
          contracted_cost = contract.get_price_for_year(current_year)
          spot_cost = aggregator.get_marginal_feedstock_price(...)

          # Blended cost
          feedstock_cost = (contract_percentage × contracted_cost
                           + (1 - contract_percentage) × spot_cost)
        ELSE:
          feedstock_cost = aggregator.get_marginal_feedstock_price(...)

      SRMC = feedstock_cost + opex + transport_cost + profit_margin

   B) Merit Order Mode (use_marginal_cost=True):
      Used for: Market price clearing (marginal producer determination)

      Feedstock Cost:
        Always use spot marginal cost (cost of next incremental tonne)
        feedstock_cost = aggregator.get_marginal_feedstock_price(...)

      SRMC = feedstock_cost + opex + transport_cost + profit_margin

   Rationale for Two Modes:
     - Market clearing: Use marginal spot price (determines who produces)
     - Financial tracking: Use blended price (actual costs paid)

   Example:
     Site with contract:
       contracted_cost = $600/t (tier 2 with escalation)
       spot_cost = $800/t (tier 3 marginal)
       contract_percentage = 0.85

     Blended mode (NPV/EBIT):
       feedstock = 0.85 × $600 + 0.15 × $800 = $630/t
       SRMC = $630 + $285.71 + $600 + $250 = $1,765.71/t

     Marginal mode (market clearing):
       feedstock = $800/t (spot marginal)
       SRMC = $800 + $285.71 + $600 + $250 = $1,935.71/t
   ```

2. **`get_contracted_capacity(current_year) -> float`**
   ```
   Purpose: Calculate feedstock covered by active contract

   Formula:
     IF active_contract AND active_contract.is_active(current_year):
       RETURN max_capacity × design_load_factor × contract_percentage
     ELSE:
       RETURN 0.0

   Example:
     max_capacity = 100,000 t/yr
     design_load_factor = 0.80
     contract_percentage = 0.85

     contracted_capacity = 100,000 × 0.80 × 0.85 = 68,000 t/yr
   ```

3. **`get_spot_capacity(current_year) -> float`**
   ```
   Purpose: Calculate feedstock purchased on spot market

   Formula:
     IF active_contract AND active_contract.is_active(current_year):
       RETURN max_capacity × design_load_factor × (1 - contract_percentage)
     ELSE:
       RETURN max_capacity × design_load_factor  # All spot if no contract

   Example (with contract):
     max_capacity = 100,000 t/yr
     design_load_factor = 0.80
     contract_percentage = 0.85

     spot_capacity = 100,000 × 0.80 × 0.15 = 12,000 t/yr

   Example (no contract):
     spot_capacity = 100,000 × 0.80 = 80,000 t/yr (100% spot)
   ```

4. **`calculate_production_output() -> float`**
   ```
   Purpose: Calculate annual production considering contracts

   Formula:
     contracted_cap = get_contracted_capacity(current_year)
     spot_cap = get_spot_capacity(current_year)

     # Apply separate load factors
     contracted_production = contracted_cap × aggregator.contracted_load_factor
     spot_production = spot_cap × aggregator.spot_load_factor

     # Apply streamday percentage
     total_production = (contracted_production + spot_production)
                        × streamday_percentage

     # Respect demand allocation if enabled
     IF model.demand_allocation:
       total_production = min(total_production, allocated_demand)

     RETURN total_production

   Example (contracted site, good year):
     contracted_cap = 68,000 t/yr
     spot_cap = 12,000 t/yr
     contracted_load_factor = 1.0 (full supply)
     spot_load_factor = 1.0 (full supply)
     streamday_percentage = 0.97
     allocated_demand = 100,000 t/yr (no constraint)

     contracted_prod = 68,000 × 1.0 = 68,000
     spot_prod = 12,000 × 1.0 = 12,000
     total = (68,000 + 12,000) × 0.97 = 77,600 t/yr

   Example (contracted site, feedstock shortage):
     contracted_load_factor = 0.7 (only 70% available)
     spot_load_factor = 0.3 (low spot availability)

     contracted_prod = 68,000 × 0.7 = 47,600
     spot_prod = 12,000 × 0.3 = 3,600
     total = (47,600 + 3,600) × 0.97 = 49,664 t/yr

   Example (oversupply, demand constrained):
     contracted_prod + spot_prod = 80,000 t/yr
     streamday = 0.97 → 77,600 t/yr potential
     allocated_demand = 50,000 t/yr (demand allocation active)

     final_production = min(77,600, 50,000) = 50,000 t/yr
   ```

**Take-or-Pay Dynamics**:
```
Contract Obligation:
  Site must PAY for contracted volume even if not producing

Scenario: Market price drops below SRMC
  market_price = $1,500/t
  SRMC = $1,600/t (unprofitable)

  Without contract:
    Rational decision: Don't produce (lose nothing)

  With contract (85% contracted):
    contracted_feedstock_cost = 68,000 t × $600/t = $40,800,000

    Option A: Don't produce
      Loss = $40,800,000 (sunk cost - must pay contract)

    Option B: Produce and sell
      Revenue = 77,600 t × $1,500/t = $116,400,000
      Costs:
        Contracted feedstock = $40,800,000 (already paid)
        Spot feedstock = 12,000 t × $800/t = $9,600,000
        Opex = 77,600 t × $285.71/t = $22,171,216
        Transport = 77,600 t × $600/t = $46,560,000
      Total costs = $119,131,216
      Loss = $116,400,000 - $119,131,216 = -$2,731,216

    Decision: Produce (lose $2.7M instead of $40.8M)

  Result: Site produces despite market_price < SRMC
          Creates "forced production" that deepens oversupply
```

---

#### 3.3.3 Investor (Enhanced)

**New Attributes**:
```python
contracts: List[FeedstockContract]           # Owned contracts
```

**Modified Methods**:

1. **`create_contract(aggregator, plant, current_year) -> FeedstockContract`**
   ```
   Purpose: Create new feedstock contract when building site

   Algorithm:
     # Sample contract coverage
     contract_percentage = random.uniform(contract_percentage_min,
                                          contract_percentage_max)
                        # Default: random.uniform(0.80, 0.90)

     # Calculate effective capacity
     annual_capacity = plant.max_capacity × plant.design_load_factor
     contract_volume = annual_capacity × contract_percentage

     # Allocate feedstock and get tier price
     initial_price = aggregator.allocate_contract(contract_volume)

     # Create contract
     duration = int(config.get("contract_duration", 20))  # 20 years default
     contract = FeedstockContract(
       contract_id=f"contract_{plant.site_id}",
       investor_id=self.investor_id,
       aggregator_id=plant.state_id,
       plant_id=plant.site_id,
       initial_contract_price=initial_price,
       start_year=current_year,
       end_year=current_year + duration,
       duration=duration,
       annual_capacity=annual_capacity,
       contract_percentage=contract_percentage,
       escalation_rate=0.03,
       status="active"
     )

     # Register contract
     self.contracts.append(contract)
     aggregator.register_contract(contract)

     RETURN contract

   Example:
     Plant: max_capacity = 100,000, design_load_factor = 0.8
     Sampled contract_percentage = 0.87

     annual_capacity = 100,000 × 0.8 = 80,000 t/yr
     contract_volume = 80,000 × 0.87 = 69,600 t/yr

     Aggregator allocation (assume cumulative = 200,000):
       Tier 2 remaining = 40,000 t @ $600/t
       Tier 3 = 29,600 t @ $800/t
       Weighted price = (40,000×$600 + 29,600×$800) / 69,600 = $680/t

     Contract created:
       initial_contract_price = $680/t
       contract_percentage = 0.87 (87%)
       duration = 20 years
       escalation_rate = 3%
   ```

2. **`calculate_NPV(...) -> float`** (Modified)
   ```
   Purpose: Calculate NPV including contract pricing

   Changes from Baseline:
     - Uses blended feedstock cost (contracted + spot)
     - Contract prices escalate at 3%/year
     - Spot prices escalate at 2%/year (market escalation)
     - After contract expires (year 21+), 100% spot pricing

   Formula:
     NPV = -CAPEX + Σ(CF_t / (1 + r)^t) for t = 1 to T

   Where CF_t depends on contract status:

   Years 1-20 (contract active):
     contracted_feedstock_cost_t = initial_price × (1.03)^t
     spot_feedstock_cost_t = initial_spot_price × (1.02)^t

     blended_feedstock_cost_t = (contract_percentage × contracted_cost_t
                                 + (1 - contract_percentage) × spot_cost_t)

     revenue_t = forecast_price_t × production
     costs_t = blended_feedstock_cost_t + opex + transport
     CF_t = production × (revenue_t - costs_t)

   Years 21+ (contract expired):
     feedstock_cost_t = spot_price × (1.02)^t

     revenue_t = forecast_price_t × production
     costs_t = feedstock_cost_t + opex + transport
     CF_t = production × (revenue_t - costs_t)

   Example:
     CAPEX = $300M
     production = 77,600 t/yr
     contract_percentage = 0.85
     initial_contracted_price = $600/t
     initial_spot_price = $800/t
     forecast_price_0 = $1,610/t
     discount_rate = 8%

     Year 1:
       contracted_cost = $600 × 1.03 = $618/t
       spot_cost = $800 × 1.02 = $816/t
       blended = 0.85 × $618 + 0.15 × $816 = $647.70/t
       revenue = $1,610 × 77,600 = $124,936,000
       costs = $647.70 × 77,600 + $285.71 × 77,600 + $600 × 77,600
         = $119,270,136
       CF_1 = $124,936,000 - $119,270,136 = $5,665,864
       PV_1 = $5,665,864 / 1.08 = $5,246,170

     ...repeat for 25 years...

     NPV = -$300M + Σ(PV_t) = [calculated value]
   ```

3. **`evaluate() -> None`** (Modified)
   ```
   Purpose: Calculate EBIT/ROACE including contract costs

   EBIT Calculation:
     total_ebit = 0.0

     FOR each owned site:
       IF operational:
         production = site.year_production_output
         revenue = production × model.market_price

         # Calculate actual feedstock cost paid
         IF site.active_contract:
           contracted_vol = site.get_contracted_capacity(...)
           spot_vol = site.get_spot_capacity(...)

           contracted_cost = contract.get_price_for_year(current_year)
           spot_cost = aggregator.get_marginal_feedstock_price(...)

           feedstock_cost = (contracted_vol × contracted_cost
                            + spot_vol × spot_cost)
         ELSE:
           feedstock_cost = production × aggregator.get_marginal_feedstock_price(...)

         other_costs = production × (opex + transport_cost)
         site_ebit = revenue - feedstock_cost - other_costs

         total_ebit += site_ebit

     ROACE = total_ebit / total_capital_invested (if > 0)

     [Discount rate adjustment logic same as baseline]

   Key Difference from Baseline:
     - Must account for blended contracted/spot costs
     - Contract escalation affects costs year-over-year
   ```

4. **`invest() -> None`** (Enhanced)
   ```
   Purpose: Evaluate and execute new site investments with contracts

   Algorithm (additions to baseline):
     ...
     [Standard site selection and NPV calculation as baseline]
     ...

     IF NPV > min_NPV_threshold:
       # Create production site
       site = SAFProductionSite(...)

       # NEW: Create feedstock contract
       contract = self.create_contract(
         aggregator=aggregator,
         plant=site,
         current_year=current_year
       )

       # Link contract to site
       site.active_contract = contract

       # Register with model
       model.all_contracts.append(contract)
       model.new_contracts_this_year.append(contract)

       # Standard registration
       self.owned_assets.append((asset_dict, site))
       model.production_sites.append(site)
       model.schedule.add(site)

       LOG investment and contract details
   ```

---

### 3.4 Model Orchestration Changes

#### 3.4.1 Initialization Additions

**New Steps**:
```
...
[Standard baseline initialization]
...

5b. Contract Creation for Initial Sites:
    FOR each initial production site:
      # Find owning investor
      investor = find_investor_by_id(site.investor_id)

      # Get aggregator
      aggregator = aggregators[site.state_id]

      # Create contract
      contract = investor.create_contract(
        aggregator=aggregator,
        plant=site,
        current_year=start_year
      )

      # Register contract
      model.all_contracts.append(contract)
      model.new_contracts_this_year.append(contract)
      aggregator.register_contract(contract)

      # Link to site
      site.active_contract = contract

      LOG contract details

5c. Calculate Initial Demand Split:
    FOR each aggregator:
      aggregator.total_contracted_demand = 0.0
      aggregator.total_spot_demand = 0.0

    FOR each operational site:
      aggregator = site.aggregator
      contracted_cap = site.get_contracted_capacity(current_year)
      spot_cap = site.get_spot_capacity(current_year)

      contracted_demand = contracted_cap × design_load_factor × streamday
      spot_demand = spot_cap × design_load_factor × streamday

      aggregator.total_contracted_demand += contracted_demand
      aggregator.total_spot_demand += spot_demand

5d. Update Supply (calculate load factors):
    FOR each aggregator:
      aggregator.update_supply()  # Calculates contracted/spot load factors

5e. Initial Production:
    FOR each operational site:
      site.produce()  # Uses load factors to calculate production

5f. Calculate Initial Spot Prices:
    FOR each state:
      spot_price = calculate_state_spot_price(
        state_id=state,
        new_contracts_this_year=new_contracts_this_year,
        aggregator=aggregator
      )
      model.state_spot_prices[state] = spot_price
      LOG spot price

[Continue with standard initialization]
```

---

#### 3.4.2 Step Sequence Modifications

**New Model Step() - Changes from Baseline**:

```
STEP(tick):

[Same as baseline: steps 1-2]

2b. CONTRACT RENEWAL CHECK (NEW)
    Purpose: Automatically renew expiring contracts

    FOR each production site:
      IF site.active_contract:
        contract = site.active_contract
        current_year = start_year + tick

        IF current_year == contract.end_year + 1:  # Contract just expired
          # Find owning investor
          investor = find_investor_by_id(site.investor_id)
          aggregator = aggregators[site.state_id]

          # Renew at same tier
          renewal_price = aggregator.renew_contract_at_same_tier(
            existing_contract=contract,
            current_year=current_year
          )

          # Create new contract
          new_contract = FeedstockContract(
            contract_id=f"contract_{site.site_id}_renewal_{current_year}",
            investor_id=site.investor_id,
            aggregator_id=site.state_id,
            plant_id=site.site_id,
            initial_contract_price=renewal_price,
            start_year=current_year,
            end_year=current_year + contract.duration,
            duration=contract.duration,
            annual_capacity=contract.annual_capacity,
            contract_percentage=contract.contract_percentage,
            escalation_rate=0.0,  # Resets (no carryover)
            status="active"
          )

          # Update references
          investor.contracts.append(new_contract)
          site.active_contract = new_contract
          aggregator.register_contract(new_contract)
          model.new_contracts_this_year.append(new_contract)

          LOG renewal

2c. CALCULATE CONTRACTED VS SPOT DEMAND (NEW)
    Purpose: Split demand for load factor calculation

    FOR each aggregator:
      aggregator.total_contracted_demand = 0.0
      aggregator.total_spot_demand = 0.0

    FOR each operational site:
      aggregator = site.aggregator
      contracted_cap = site.get_contracted_capacity(current_year)
      spot_cap = site.get_spot_capacity(current_year)

      # Apply design factors
      contracted_demand = contracted_cap × design_load_factor × streamday
      spot_demand = spot_cap × design_load_factor × streamday

      aggregator.total_contracted_demand += contracted_demand
      aggregator.total_spot_demand += spot_demand

[Step 3: PRODUCE - same as baseline]

3b. ALLOCATE DEMAND (NEW, if enabled)
    Purpose: When oversupply, allocate demand proportionally

    current_year = start_year + tick
    total_demand = get_saf_demand_forecast(current_year, ...)

    IF config.get('enable_demand_allocation', True):
      demand_allocation = allocate_demand_to_sites(
        production_sites=operational_sites,
        total_demand=total_demand,
        current_year=current_year
      )

      FOR each site:
        site.allocated_demand = demand_allocation[site.site_id]
    ELSE:
      FOR each site:
        site.allocated_demand = None  # No constraint

    Allocation Algorithm:
      total_contracted = sum(site.get_contracted_capacity() for all sites)
      total_spot = sum(site.get_spot_capacity() for all sites)
      total_capacity = total_contracted + total_spot

      IF total_capacity > total_demand:
        # Oversupply - allocate proportionally
        FOR each site:
          site_fraction = (site.contracted + site.spot) / total_capacity
          site.allocated_demand = total_demand × site_fraction
      ELSE:
        # Undersupply - no constraints
        FOR each site:
          site.allocated_demand = None

[Step 4: UPDATE MARKET PRICE - same logic as baseline]

[Step 5: UPDATE PRICE FORECAST]
    Note: Forecast now includes contracted sites with blended costs
    (Implementation details same as baseline, but SRMC calculations
     now use contract-aware feedstock costs)

5b. CALCULATE STATE SPOT PRICES (NEW)
    Purpose: Track spot market prices per state

    FOR each state:
      spot_price = calculate_state_spot_price(
        state_id=state,
        new_contracts_this_year=new_contracts_this_year,
        previous_spot_price=state_spot_prices.get(state),
        aggregator=aggregator
      )
      model.state_spot_prices[state] = spot_price

    Spot Price Logic:
      IF new contracts signed this year in this state:
        # Use weighted average of new contract prices
        spot_price = weighted_avg(new_contract_prices)
      ELIF previous_spot_price exists:
        # Apply market escalation
        spot_price = previous_spot_price × (1 + market_escalation_rate)
      ELSE:
        # Fallback to tier 1 price
        spot_price = aggregator.tier_1_cost

[Steps 6-10: Same as baseline]
```

---

### 3.5 Configuration Parameters (New Model)

**Additional/Modified Parameters**:
```
Parameter                       Default     Unit            Description
------------------------------------------------------------------------------------
# Contract Parameters (NEW)
contract_percentage_min         0.80        fraction        Min contract coverage
contract_percentage_max         0.90        fraction        Max contract coverage
contract_escalation_rate        0.03        fraction/year   Annual contract escalation

# Tier System Parameters (NEW)
tier_capacity_size              120,000     tonnes/year     Capacity per tier
tier_1_cost                     400.0       USD/tonne       Base tier price
tier_cost_increment             200.0       USD/tonne       Price increase per tier
aggregator_profit_margin        50.0        USD/tonne       Aggregator margin

# Market Escalation (NEW)
inflation_rate                  0.02        fraction/year   CPI inflation (disabled in baseline)
tech_improvement_rate           0.0         fraction/year   Tech cost reduction (disabled)
market_escalation_rate          0.02        fraction/year   Net market escalation (2%)

# Modified Parameters
Optimism_factor_sample_min      1.0         multiplier      Price optimism (no change)
Optimism_factor_sample_max      1.0         multiplier      Price optimism (no change)
```

---

### 3.6 Key Behavioral Differences

#### 3.6.1 Investment Timing

**Baseline**:
```
All sites have identical feedstock costs within a state
→ No first-mover advantage
→ Investment timing driven purely by NPV signals
```

**New Model**:
```
Early investors get lower tier prices
→ Strong first-mover advantage
→ Creates "land rush" dynamics when NPVs turn positive
→ Late entrants face higher costs, may be uncompetitive

Example Timeline:
  Year 5: Investor A builds site
    Contract @ tier 1 ($400/t) → SRMC = $1,535.71/t

  Year 8: Investor B builds site
    Contract @ tier 3 ($800/t) → SRMC = $1,935.71/t
    ($400/t disadvantage)

  Year 15: Market price = $1,700/t
    Investor A: Profit = $164.29/t
    Investor B: Loses money (SRMC > price), exits market
```

---

#### 3.6.2 Market Volatility

**Baseline**:
```
Sites respond flexibly to price signals
→ Low price → Sites shut down (no production)
→ Oversupply resolves quickly
→ Prices stabilize near average SRMC
```

**New Model**:
```
Contracts create commitment
→ Low price → Sites produce at loss (contract obligation)
→ Oversupply persists longer
→ Deeper price crashes, boom-bust cycles

Feedback Loop:
  1. High prices → New investments → Oversupply
  2. Prices crash below average SRMC
  3. Baseline: Sites stop → Supply falls → Prices recover
     New: Contracted sites keep producing → Supply stays high → Prices stay low
  4. Eventually, sites without contracts exit
  5. Supply finally contracts, prices recover
  6. Repeat
```

---

#### 3.6.3 Investor Risk

**Baseline**:
```
Pure spot market risk
→ Revenue risk: Market price volatility
→ Cost risk: None (feedstock price fixed)
→ Flexibility: Can stop production anytime
```

**New Model**:
```
Contract-locked risk profile
→ Revenue risk: Market price volatility (same)
→ Cost risk: Contract escalation (3%/year) vs market (2%/year)
→ Commitment risk: Must pay contract even if not producing
→ Upside: Price certainty for 80-90% of feedstock

Risk Example:
  Scenario: Feedstock shortage

  Baseline site:
    Spot price jumps $400 → $800 (100% increase)
    All costs double → Heavy losses

  Contracted site:
    Contract: 85% @ $620 (escalated from $600)
    Spot: 15% @ $800 (doubled)
    Blended: 0.85×$620 + 0.15×$800 = $647
    Cost increase: $447 → $647 (45% increase)
    → Partially hedged
```

---

#### 3.6.4 Market Structure Evolution

**Baseline**:
```
Homogeneous competitors
→ All sites in same state have equal costs
→ Exit/entry based purely on recent returns
→ No persistent cost advantages
```

**New Model**:
```
Stratified market structure
→ Early sites: Low-tier contracts → Low costs → High profitability
→ Late sites: High-tier contracts → High costs → Marginal/unprofitable
→ Exit during downturns: Late entrants exit first
→ Persistent leaders: Early sites maintain advantage indefinitely

Market Concentration:
  Over time, market becomes dominated by early investors
  → Late entrants struggle to compete
  → Barriers to entry increase as tiers fill
  → "Winner-take-most" dynamics
```

---

## 4. Comparative Analysis

### 4.1 Feedstock Procurement

| Aspect | Baseline | New Model |
|--------|----------|-----------|
| **Pricing** | Fixed per state | Tiered (supply curve) |
| **Contracts** | None (100% spot) | 20-year contracts (80-90%) |
| **Cost Predictability** | None | High (contracted volume) |
| **Price Range** | $400-$800/t (state variation) | $400-$1200+/t (tier variation) |
| **Escalation** | None | 3%/year (contracts), 2%/year (spot) |
| **Flexibility** | Full (buy what you need) | Partial (contracted + spot) |
| **First-Mover Advantage** | None | Strong (lower tiers) |
| **Risk Profile** | Pure price risk | Blended (commitment + price) |

### 4.2 Production Decisions

| Aspect | Baseline | New Model |
|--------|----------|-----------|
| **When Unprofitable** | Shut down (produce 0) | Often produce (contract obligation) |
| **Production Constraint** | Feedstock + demand | Feedstock + demand + allocation |
| **Cost Calculation** | Simple (spot price) | Complex (blended contracted/spot) |
| **Load Factors** | Single (annual_load_factor) | Dual (contracted/spot factors) |
| **Oversupply Response** | Immediate shutdown | Delayed (contracts force production) |
| **Profit Margins** | Responsive to signals | Dampened by contracts |

### 4.3 Investment Behavior

| Aspect | Baseline | New Model |
|--------|----------|-----------|
| **NPV Calculation** | Spot prices throughout | Blended (contracts years 1-20, spot 21+) |
| **Investment Trigger** | NPV > 0 | NPV > 0 (same threshold) |
| **Timing Incentive** | Weak | Strong (tier competition) |
| **Location Choice** | Lowest spot price | Lowest available tier |
| **Capital At Risk** | CAPEX only | CAPEX + contract commitment |
| **Expected Returns** | Volatile (pure spot) | Stabilized (contracts hedge) |
| **Investor Differentiation** | Entry timing only | Entry timing + tier position |

### 4.4 Market Dynamics

| Aspect | Baseline | New Model |
|--------|----------|-----------|
| **Price Volatility** | Moderate | Higher (sticky supply) |
| **Boom-Bust Cycles** | Short, shallow | Long, deep |
| **Market Clearing** | Efficient (flexible supply) | Inefficient (forced production) |
| **Competition** | Symmetric (equal costs) | Asymmetric (tier advantages) |
| **Barriers to Entry** | Low (CAPEX only) | Rising (higher tiers) |
| **Market Concentration** | Even | Increasing (early winners dominate) |
| **Long-Run Equilibrium** | Near average SRMC | Below contracted SRMC (oversupply bias) |

### 4.5 Simulation Outcomes

**Typical Baseline Run (100 years)**:
```
Price Pattern:
  Years 1-20: Steady rise ($1,500 → $1,800) as demand grows
  Years 21-40: Volatility increases (±$200 swings) as capacity catches up
  Years 41-100: Stabilizes near $1,600-$1,700 (average SRMC)

Investment Pattern:
  Early: Slow, cautious (1-2 sites/decade)
  Middle: Accelerates (3-5 sites/decade)
  Late: Episodic bursts (boom-bust)

Market Structure:
  Moderate concentration
  Investors 5-15 range over time
  Sites 10-50 range
  Turnover: Some investors exit, new enter
```

**Typical New Model Run (100 years)**:
```
Price Pattern:
  Years 1-20: Rapid rise ($1,500 → $2,000+) then crash
  Years 21-50: Severe boom-bust cycles ($1,200-$2,000 swings)
  Years 51-100: Gradual stabilization, but with persistent volatility

Investment Pattern:
  Years 1-10: Land rush (8-12 sites built, racing for low tiers)
  Years 11-30: Pause (tiers 1-3 filled, tier 4+ uneconomic)
  Years 31-60: Episodic (only when prices spike above tier 4+ costs)
  Years 61-100: Steady replacement (renewals dominate)

Market Structure:
  High concentration
  Early investors (years 1-10) dominate
  Sites: Initially 10-15, expands to 20-30, then contracts to 12-18 survivors
  Low turnover: Early sites maintain advantages, late sites exit

Tier Allocation:
  Year 100 snapshot (Tamil Nadu state):
    Tier 1: Filled (3-4 contracts from years 1-5)
    Tier 2: Filled (4-5 contracts from years 5-12)
    Tier 3: Partial (2-3 contracts from years 12-25)
    Tier 4+: Empty (uncompetitive, no contracts)
```

---

## 5. Configuration Parameters

### 5.1 Complete Parameter Reference

**Economic Parameters**:
```python
min_NPV_threshold: 0.0                # USD - Minimum NPV to invest
opex: 285.71                          # USD/tonne - Operating expenditure
transport_cost: 600.0                 # USD/tonne - Feedstock transport
profit_margin: 250.0                  # USD/tonne - Profit margin per tonne
capex_total_cost: 300,000,000         # USD - Total plant CAPEX
atf_plus_price: 2000.0                # USD/tonne - SAF price cap
```

**Capacity Parameters**:
```python
max_capacity: 100,000                 # tonnes/year - Plant nameplate capacity
streamday_min: 0.95                   # fraction - Min plant uptime
streamday_max: 0.98                   # fraction - Max plant uptime
```

**Investor Parameters**:
```python
initial_num_investors: 1              # count - Starting investors
DR_sample_min: 0.06                   # fraction - Min discount rate (6%)
DR_sample_max: 0.12                   # fraction - Max discount rate (12%)
DR_min: 0.04                          # fraction - DR lower bound (4%)
DR_max: 0.20                          # fraction - DR upper bound (20%)
DR_target: 0.075                      # fraction - Target DR (7.5%)
DR_sensitivity_parameter: 0.2         # - DR adjustment speed
ideal_roace: 0.075                    # fraction - Target ROACE (7.5%)
ROACE_stability_min: 0.06             # fraction - Lower stability band (6%)
ROACE_stability_max: 0.09             # fraction - Upper stability band (9%)
Investment_horizon_length: 25         # years - NPV horizon
Optimism_factor_sample_min: 1.0       # multiplier - Min price optimism
Optimism_factor_sample_max: 1.0       # multiplier - Max price optimism
```

**Site Parameters**:
```python
initial_num_SAF_sites: 1              # count - Starting sites
saf_plant_construction_time: 4        # years - Construction duration
```

**Feedstock Parameters (Baseline)**:
```python
feedstock_multiplier_min: 1.0         # multiplier - Min supply stochasticity
feedstock_multiplier_max: 1.0         # multiplier - Max supply stochasticity
```

**Feedstock Parameters (New Model)**:
```python
# Tier System
tier_capacity_size: 120,000           # tonnes/year - Capacity per tier
tier_1_cost: 400.0                    # USD/tonne - Base tier price
tier_cost_increment: 200.0            # USD/tonne - Price increase per tier
aggregator_profit_margin: 50.0        # USD/tonne - Aggregator margin

# Contracts
contract_percentage_min: 0.80         # fraction - Min contract coverage (80%)
contract_percentage_max: 0.90         # fraction - Max contract coverage (90%)
contract_escalation_rate: 0.03        # fraction/year - Annual escalation (3%)
```

**Cost Evolution**:
```python
capex_annual_decrease: 0.01           # fraction/year - CAPEX learning rate
inflation_rate: 0.0                   # fraction/year - CPI inflation (disabled)
tech_improvement_rate: 0.0            # fraction/year - Tech improvement (disabled)
market_escalation_rate: 0.02          # fraction/year - Net market escalation (2%, new model only)
```

**Time Parameters**:
```python
start_year: 2024                      # year - Simulation start
blending_mandate: 0.01                # fraction - SAF mandate (not enforced)
```

**Boolean Flags**:
```python
operational_initially: True           # Sites start operational (no construction delay)
capex_decrease: False                 # Enable CAPEX learning curve
enable_demand_allocation: True        # Allocate demand when oversupply (new model only)
```

---

### 5.2 States Data Structure

**Format** (CSV: `states_data_{feedstock_scenario}.csv`):
```
state,max_supply (Tonnes/year),feedstock_price(USD/Tonne),feedstock_type
TAMIL_NADU,600000,400,UCO
KARNATAKA,500000,450,UCO
TELANGANA,550000,425,UCO
...
```

**Example States**:
```
State           Max Supply (t/yr)   Baseline Price (USD/t)   Feedstock Type
--------------------------------------------------------------------------------
TAMIL_NADU      600,000             400                      UCO
KARNATAKA       500,000             450                      UCO
TELANGANA       550,000             425                      UCO
MAHARASHTRA     700,000             380                      UCO
GUJARAT         650,000             410                      UCO
WEST_BENGAL     450,000             460                      UCO
UTTAR_PRADESH   800,000             395                      UCO
RAJASTHAN       520,000             440                      UCO
PUNJAB          580,000             420                      UCO
HARYANA         540,000             430                      UCO
```

**Feedstock Scenarios**:
- **Oversupply**: High max_supply values, low scarcity
- **Balanced**: Moderate max_supply, realistic constraints
- **Scarcity**: Low max_supply, high competition

---

## 6. Data Structures

### 6.1 Core Classes

#### FeedstockAggregator
```python
class FeedstockAggregator(Agent):
    # Identity
    state_id: str

    # Supply characteristics
    max_supply: float
    feedstock_price: float  # Baseline only
    current_supply: float
    annual_load_factor: float
    available_feedstock: float

    # Stochasticity
    multiplier_min: float
    multiplier_max: float

    # New Model additions
    tier_capacity_size: float
    tier_1_cost: float
    tier_cost_increment: float
    tiers: List[Tuple[float, float, float]]
    aggregator_profit_margin: float
    cumulative_allocated: float
    contracts: List[FeedstockContract]
    total_contracted_demand: float
    total_spot_demand: float
    contracted_load_factor: float
    spot_load_factor: float
```

#### SAFProductionSite
```python
class SAFProductionSite(Agent):
    # Identity
    site_id: str
    state_id: str
    investor_id: str

    # Capacity
    max_capacity: float
    design_load_factor: float

    # Costs
    opex: float
    transport_cost: float
    profit_margin: float
    srmc: float

    # References
    aggregator: FeedstockAggregator

    # Construction
    capex_schedule: List[float]
    construction_years: int
    tick_built: int
    operational_year: int

    # Production
    streamday_percentage: float
    year_production_output: float

    # New Model additions
    active_contract: Optional[FeedstockContract]
    allocated_demand: float
```

#### Investor
```python
class Investor(Agent):
    # Identity
    investor_id: str

    # Financial characteristics
    discount_rate: float
    optimism_factor: float
    total_capital_invested: float

    # Portfolio
    owned_assets: List[Tuple[Dict, SAFProductionSite]]
    num_owned_assets: int

    # Strategy
    capex_schedule: List[float]
    min_NPV_threshold: float
    investment_horizon: int

    # Performance tracking
    inv_ebit_history: Deque[float]
    roace_history: Deque[float]

    # Market context
    current_tick: int
    states_data: Dict[str, Dict]
    consumer_price_forecast: List[float]

    # New Model additions
    contracts: List[FeedstockContract]
```

#### FeedstockContract (New Model)
```python
@dataclass
class FeedstockContract:
    # Identity
    contract_id: str
    investor_id: str
    aggregator_id: str
    plant_id: str

    # Pricing
    initial_contract_price: float
    escalation_rate: float

    # Timing
    start_year: int
    end_year: int
    duration: int

    # Volume
    annual_capacity: float
    contract_percentage: float

    # Status
    status: str  # "active" | "expired"

    # Properties
    @property
    def contracted_volume() -> float
    @property
    def spot_volume() -> float

    # Methods
    def get_price_for_year(current_year: int) -> float
    def is_active(current_year: int) -> bool
    def expire() -> None
```

---

### 6.2 Model State

```python
class SAFMarketModel(Model):
    # Core components
    schedule: RandomActivation
    datacollector: DataCollector

    # Configuration
    config: dict
    booleans: dict
    states_data: dict
    atf_demand_forecast: dict

    # Agents
    aggregators: Dict[str, FeedstockAggregator]
    investors: List[Investor]
    production_sites: List[SAFProductionSite]

    # Market state
    states_available_feedstock: Dict[str, float]
    market_price: float
    demand: float
    marginal_details: dict
    consumer_price_forecast: List[float]

    # New Model additions
    all_contracts: List[FeedstockContract]
    new_contracts_this_year: List[FeedstockContract]
    state_spot_prices: Dict[str, float]
    demand_allocation: Optional[Dict[str, float]]
```

---

## 7. Summary

This document provides a complete technical specification of both the baseline and new SAF Market Models. Key takeaways:

**Baseline Model**:
- Simple, transparent spot market
- Symmetric competition
- Flexible production responses
- Price volatility driven by demand shocks

**New Model**:
- Realistic commodity market with contracts
- Asymmetric competition (tier advantages)
- Commitment-driven production
- Price volatility amplified by contract obligations

Both models simulate complex market dynamics emerging from individual agent decisions, providing insights into SAF market development under different regulatory and contractual regimes.

---

**End of Document**
