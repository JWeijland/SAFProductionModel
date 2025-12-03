from pyexpat import model
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from src.Agents.Feedstock_Aggregator import FeedstockAggregator
from src.Agents.SAF_Production_Site import SAFProductionSite
from src.Agents.Investor import Investor
# CLAUDE START - Import for Phase 1 contract implementation
from src.Agents.FeedstockContract import FeedstockContract
# CLAUDE END - Import for Phase 1 contract implementation
import random
from src.utils import (
    calculate_consumer_price,
    forecast_consumer_prices,
    find_operational_sites,
    get_saf_demand_forecast,
    year_for_tick,
    # CLAUDE START - Import contract pricing functions
    calculate_state_spot_price,
    # CLAUDE END - Import contract pricing functions
)
import logging
import os
import pandas as pd
import numpy as np
 
logger = logging.getLogger("Model")
 
class SAFMarketModel(Model):
    """
    SAFMarketModel encapsulates the agent-based simulation of a Sustainable Aviation Fuel market.
 
    The model coordinates:
      - Feedstock aggregators (state-level feedstock availability and pricing)
      - SAF production sites (capacity build-out, operational status, production decisions)
      - Investors (capital allocation, investment evaluation, expansion logic)
      - Market clearing (merit-order price formation)
 
    The model uses Mesa's RandomActivation scheduler. Model and agent-level metrics are controlled via step, and recorded for each tick where agents are scheduled to act in a specific order.
 
    Core simulation loop stages (per tick):
      - update_supply (agents adjust internal supply state)
      - produce (operational sites produce SAF)
      - calculate (market price is determined)
      - evaluate (investors reassess asset performance)
      - invest (investors may add new sites)
 
    The model then introduces a new potential investor, and lets them evaluate investment opportunities or leave model.
    """
 
    def __init__(
        self, config: dict, states_data: dict, atf_demand_forecast: dict, booleans: dict, seed = None
    ):
        """
        Initialise the model state, create agents, generate initial forecasts, and register them with the scheduler.
 
        Side Effects:
            - Builds a RandomActivation scheduler.
            - Instantiates FeedstockAggregator agents for each state.
            - Creates initial Investor agents.
            - Creates initial SAFProductionSite agents and assigns them to investors.
            - Generates and broadcasts an initial consumer price forecast to investors.
            - Collects an initial DataCollector snapshot at tick 0.
        """
        super().__init__()
 
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
 
        self.config = config
        self.states_data = states_data
        self.booleans = booleans
        self.atf_demand_forecast = atf_demand_forecast
 
        self.datacollector = DataCollector(
            model_reporters={
                "Tick": lambda m: (
                    int(getattr(m, "schedule").time)
                    if getattr(m, "schedule", None) is not None
                    else None
                ),
                "Year": lambda m: (
                    year_for_tick(int(m.config["start_year"]), int(m.schedule.time))
                    if getattr(m, "schedule", None) is not None
                    else None
                ),
                "Consumer_Price": lambda m: getattr(m, "market_price", None),
                "Market_Price": lambda m: getattr(m, "market_price", None),
                "Demand": lambda m: getattr(m, "demand", None),
                # CLAUDE - MARKET PRICE FIX: Total_Capacity shows TRUE capacity (like copy_0)
                # This is the maximum production capability considering feedstock availability
                # Formula: max_capacity × design_load_factor × annual_load_factor
                #
                # NOTE: EXCLUDES streamday_percentage (following copy_0 principle)
                # Streamday represents operational inefficiency (maintenance, breakdowns), not capacity constraint.
                # Market capacity should reflect what's technically possible with available feedstock,
                # not what's achieved with operational losses.
                #
                # This is used for "Capacity vs Demand" graph and market price calculation.
                # Actual production (with streamday) is ~30% lower and tracked in Actual_Production metric.
                "Total_Capacity": lambda m: sum(
                    (site.max_capacity * site.design_load_factor *
                     site.aggregator.annual_load_factor)
                    for site in getattr(m, "production_sites", [])
                    if site.operational_year <= m.schedule.time
                ),
                # CLAUDE - TAKE-OR-PAY: Actual_Production shows REALIZED production (respects demand allocation)
                # This is always <= demand when allocation is enabled
                "Actual_Production": lambda m: sum(
                    (
                        site.year_production_output.get(m.schedule.time, 0.0)
                        if isinstance(site.year_production_output, dict)
                        else site.year_production_output
                    )
                    for site in getattr(m, "production_sites", [])
                ),
                "Num_Investors": lambda m: len(getattr(m, "investors", [])),
                "Num_Production_Sites": lambda m: len(
                    getattr(m, "production_sites", [])
                ),
                # CLAUDE START - Contract metrics for Phase 1 implementation
                "Num_Active_Contracts": lambda m: len([
                    c for c in getattr(m, "all_contracts", [])
                    if c.is_active(year_for_tick(
                        int(m.config["start_year"]),
                        int(m.schedule.time)
                    ))
                ]),
                "Total_Contracted_Capacity": lambda m: sum(
                    c.contracted_volume
                    for c in getattr(m, "all_contracts", [])
                    if c.is_active(year_for_tick(
                        int(m.config["start_year"]),
                        int(m.schedule.time)
                    ))
                ),
                # CLAUDE END - Contract metrics for Phase 1 implementation
            },
            agent_reporters={
                # Prefer agent.current_tick; fall back to agent.tick or scheduler time
                "Tick": lambda a: (
                    int(
                        getattr(
                            a,
                            "current_tick",
                            getattr(a, "tick", int(a.model.schedule.time)),
                        )
                    )
                    if getattr(a, "model", None) and getattr(a.model, "schedule", None)
                    else getattr(a, "current_tick", getattr(a, "tick", None))
                ),
                "Year": lambda a: (
                    year_for_tick(
                        int(a.model.config["start_year"]),
                        int(
                            getattr(
                                a,
                                "current_tick",
                                getattr(a, "tick", int(a.model.schedule.time)),
                            )
                        ),
                    )
                    if getattr(a, "model", None) and getattr(a.model, "config", None)
                    else None
                ),
                "Type": lambda a: type(a).__name__,
                "Unique_ID": lambda a: a.unique_id,
                "State_ID": lambda a: getattr(a, "state_id", None),
                "Investor_ID": lambda a: getattr(a, "investor_id", None),
                "Production_Output": lambda a: (
                    getattr(a, "year_production_output", None)
                    if hasattr(a, "year_production_output")
                    else getattr(a, "tick_production_output", None)
                ),
                # CLAUDE - Contract vs Spot production tracking
                "Contracted_Production": lambda a: (
                    getattr(a, "contracted_production", 0.0)
                    if hasattr(a, "contracted_production")
                    else 0.0
                ),
                "Spot_Production": lambda a: (
                    getattr(a, "spot_production", 0.0)
                    if hasattr(a, "spot_production")
                    else 0.0
                ),
                # CLAUDE - Take-or-Pay metrics for curtailment visualization
                "Curtailed_Volume": lambda a: (
                    getattr(a, "curtailed_volume", 0.0)
                    if hasattr(a, "curtailed_volume")
                    else 0.0
                ),
                "Take_Or_Pay_Penalty": lambda a: (
                    getattr(a, "take_or_pay_penalty", 0.0)
                    if hasattr(a, "take_or_pay_penalty")
                    else 0.0
                ),
                "SRMC": lambda a: (
                    getattr(a, "srmc", None) if hasattr(a, "srmc") else None
                ),
                "Design_Load_Factor": lambda a: (
                    getattr(a, "design_load_factor", None)
                    if hasattr(a, "design_load_factor")
                    else None
                ),
                "Annual_Load_Factor": lambda a: (
                    getattr(a, "annual_load_factor", None)
                    if hasattr(a, "annual_load_factor")
                    else None
                ),
                "NPV": lambda a: getattr(a, "npv", None) if hasattr(a, "npv") else None,
                "Discount_Rate": lambda a: (
                    getattr(a, "discount_rate", None)
                    if hasattr(a, "discount_rate")
                    else None
                ),
                "ROACE": lambda a: (
                    getattr(a, "roace", None) if hasattr(a, "roace") else None
                ),
                "Feedstock_Price": lambda a: (
                    getattr(a, "feedstock_price", None)
                    if hasattr(a, "feedstock_price")
                    else None
                ),
                "Available_Feedstock": lambda a: (
                    getattr(a, "available_feedstock", None)
                    if hasattr(a, "available_feedstock")
                    else None
                ),
                "Current_Supply": lambda a: (
                    getattr(a, "current_supply", None)
                    if hasattr(a, "current_supply")
                    else None
                ),
                "Optimism_Factor": lambda a: (
                    getattr(a, "optimism_factor", None) if hasattr(a, "optimism_factor") else None
                ),
                "SAF_Demand": lambda a: (
                    getattr(a, "saf_demand", None) if hasattr(a, "saf_demand") else None
                ),
                "Consumer_Price_Forecast": lambda a: (
                    getattr(a, "consumer_price_forecast", None)
                    if hasattr(a, "consumer_price_forecast")
                    else None
                ),
                "EBIT": lambda a: (
                    getattr(a, "ebit", None) if hasattr(a, "ebit") else None
                ),
                "Raw_ROACE": lambda a: (
                    getattr(a, "raw_roace", None) if hasattr(a, "raw_roace") else None
                ),
                "Max_Supply": lambda a: (
                    getattr(a, "max_supply", None) if hasattr(a, "max_supply") else None
                ),
                # CLAUDE - Tier allocation tracking
                "Cumulative_Allocated": lambda a: (
                    getattr(a, "cumulative_allocated", None) if hasattr(a, "cumulative_allocated") else None
                ),
                "Num_Owned_Sites": lambda a: (
                    getattr(a, "num_owned_assets", None)
                    if hasattr(a, "num_owned_assets")
                    else 0
                ),
                # CLAUDE START - Contract metrics for agents (Phase 1 implementation)
                # For FeedstockAggregator agents
                "State_Spot_Price": lambda a: (
                    a.model.state_spot_prices.get(a.state_id)
                    if hasattr(a, "state_id")
                    and hasattr(a.model, "state_spot_prices")
                    else None
                ),
                "Aggregator_Contracted_Capacity": lambda a: (
                    a.get_contracted_capacity(year_for_tick(
                        int(a.model.config["start_year"]),
                        int(a.model.schedule.time)
                    ))
                    if hasattr(a, "get_contracted_capacity")
                    else None
                ),
                # Contract price for SAF Production Sites
                "Contract_Price": lambda a: (
                    a.active_contract.get_price_for_year(year_for_tick(
                        int(a.model.config["start_year"]),
                        int(a.model.schedule.time)
                    ))
                    if hasattr(a, "active_contract") and a.active_contract is not None
                    else None
                ),
                "Initial_Contract_Price": lambda a: (
                    a.active_contract.initial_contract_price
                    if hasattr(a, "active_contract") and a.active_contract is not None
                    else None
                ),
                # For Investor agents
                "Num_Contracts": lambda a: (
                    len(getattr(a, "contracts", []))
                    if hasattr(a, "contracts")
                    else None
                ),
                "Avg_Contract_Coverage": lambda a: (
                    (sum(c.contract_percentage for c in a.contracts) / len(a.contracts))
                    if hasattr(a, "contracts") and len(a.contracts) > 0
                    else None
                ),
                # CLAUDE END - Contract metrics for agents (Phase 1 implementation)
            },
        )
 
        self.schedule = RandomActivation(self)
 
        logging.info("Initialising Feedstock Aggregators...")
        self.aggregators = {}
        for state_id in states_data:
            aggregator = FeedstockAggregator(
                unique_id=f"agg_{state_id}",
                model=self,
                state_id=state_id,
                states_data=states_data,
            )
            self.aggregators[state_id] = aggregator
            self.schedule.add(aggregator)
 
        logging.info("Creating initial investors...")
        self.investors = []
        for _ in range(int(config["initial_num_investors"])):
            investor = Investor(
                unique_id=Investor.generate_investor_id(),
                model=self,
                current_tick=0,
                states_data=states_data,
                total_capital_invested=(
                    config["capex_total_cost"]
                    if booleans["operational_initially"]
                    else None
                ),
            )
            investor.num_owned_assets = 0
            self.investors.append(investor)
            self.schedule.add(investor)
 
        logging.info("Creating initial sites and assigning to investors...")
        self.production_sites = []
        for i in range(int(config["initial_num_SAF_sites"])):
            self.states_available_feedstock = {
                state_id: aggregator.max_supply
                for state_id, aggregator in self.aggregators.items()
            }

            # Only operational plants pledge feedstock
            for site in self.production_sites:
                if site.operational_year <= 0:  # During init, check against 0
                    pledged_feedstock = site.max_capacity * site.design_load_factor
                    self.states_available_feedstock[site.state_id] = max(
                        0, self.states_available_feedstock[site.state_id] - pledged_feedstock
                    )
 
            for aggregator in self.aggregators.values():
                aggregator.available_feedstock = self.states_available_feedstock[
                    aggregator.state_id
                ]
 
            investor = self.investors[i % len(self.investors)]
            state_id = random.choice(list(states_data.keys()))
            aggregator = self.aggregators[state_id]
            total_cost = (
                int(config["capex_total_cost"])
                if not booleans["capex_decrease"]
                else investor.get_dynamic_capex(
                    int(config["capex_total_cost"]),
                    self.schedule.time,
                    config["capex_annual_decrease"],
                )
            )
            construction_time = int(config["saf_plant_construction_time"])
            capex_schedule = [total_cost / construction_time] * construction_time
            operational = bool(booleans["operational_initially"])
            design_load_factor = min(
                1, aggregator.available_feedstock / self.config["max_capacity"])
            site = SAFProductionSite(
                unique_id=SAFProductionSite.generate_site_id(state_id),
                model=self,
                state_id=state_id,
                investor_id=investor.investor_id,
                config=config,
                design_load_factor=design_load_factor,
                aggregator=aggregator,
                capex_schedule=capex_schedule if operational == False else [],
            )
            if operational:
                site.operational_year = 0  # Initial plants operational at tick 0
            self.production_sites.append(site)
            self.schedule.add(site)
 
        logging.info("Generating price forecast...")
        self.consumer_price_forecast = self.generate_price_forecast()
        logging.info(f"Price forecast: {self.consumer_price_forecast}")
 
        for investor in self.investors:
            investor.consumer_price_forecast = self.consumer_price_forecast
 
        logging.info("Creating asset information for investors...")
        for site in self.production_sites:
            investor = next(
                i
                for i in self.investors
                if i.investor_id
                == site.investor_id  # finds production sites the investor owns.
            )
            if operational == True:
                investor.total_capital_invested = total_cost
            asset = investor.evaluate_investment(
                site, current_tick=self.schedule.time, operational=operational
            )
            investor.owned_assets.append((asset, site))
            investor.num_owned_assets += 1
            investor.consumer_price_forecast = self.consumer_price_forecast
 
        for investor in self.investors:
            logging.info(
                f"Investor {investor.investor_id} discount rate: {investor.discount_rate:.4f}"
            )
            for asset, _ in investor.owned_assets:
                logging.info(f"  Asset {asset['site_id']}")

        # Contract tracking attributes
        self.all_contracts = []
        self.state_spot_prices = {}
        self.new_contracts_this_year = []

        # Create contracts for initial sites
        logging.info("Creating initial feedstock contracts...")
        for site in self.production_sites:
            # Find the investor that owns this site
            investor = next(
                i for i in self.investors
                if i.investor_id == site.investor_id
            )

            # Get the aggregator for this site's state
            aggregator = self.aggregators[site.state_id]

            # Create contract (year 0)
            current_year = int(self.config["start_year"])
            contract = investor.create_contract(
                aggregator=aggregator,
                plant=site,
                current_year=current_year
            )

            # Register contract
            self.all_contracts.append(contract)
            self.new_contracts_this_year.append(contract)
            aggregator.register_contract(contract)

            # Link contract to site (enables priority allocation)
            site.active_contract = contract

            logging.info(
                f"Initial contract {contract.contract_id}: "
                f"{contract.contract_percentage:.1%} @ ${contract.initial_contract_price:.2f}/tonne"
            )

        # Calculate initial demand (needed for load factors)
        from src.utils import year_for_tick
        current_year = year_for_tick(int(config["start_year"]), 0)

        # Ensure scheduler time is set for initial calculations
        if not hasattr(self.schedule, 'time'):
            self.schedule.time = 0

        for aggregator in self.aggregators.values():
            aggregator.total_contracted_demand = 0.0
            aggregator.total_spot_demand = 0.0

        for site in self.production_sites:
            if site.operational_year <= 0:  # Check against 0 explicitly for init
                aggregator = site.aggregator
                contracted_cap = site.get_contracted_capacity(current_year)
                spot_cap = site.get_spot_capacity(current_year)

                contracted_demand = contracted_cap * site.design_load_factor * site.streamday_percentage
                spot_demand = spot_cap * site.design_load_factor * site.streamday_percentage

                aggregator.total_contracted_demand += contracted_demand
                aggregator.total_spot_demand += spot_demand

        # Update supply (calculates load factors)
        for aggregator in self.aggregators.values():
            aggregator.update_supply()

        # Initial production calculation (after contracts and load factors are set)
        logging.info("Calculating initial production for operational sites...")
        for site in self.production_sites:
            if site.operational_year <= 0:  # Check against 0 explicitly for init
                site.produce()
                logging.info(
                    f"{site.site_id}: Initial production = {site.year_production_output:,.0f} tonnes/year"
                )

        # Calculate initial spot prices
        for state_id, aggregator in self.aggregators.items():
            self.state_spot_prices[state_id] = calculate_state_spot_price(
                state_id=state_id,
                new_contracts_this_year=self.new_contracts_this_year,
                aggregator=aggregator
            )
            logging.info(
                f"Initial spot price for {state_id}: "
                f"${self.state_spot_prices[state_id]:.2f}/tonne"
            )
        # CLAUDE END - Create contracts for initial sites

        # Collect initial snapshot at tick 0
        self.datacollector.collect(self)
 
    def step(self):
        """
        Advance the simulation by one tick.
 
        Sequence:
            1) Recompute available feedstock after existing site commitments.
            2) Agents update_supply() to refresh internal supply state.
            3) Agents produce() to generate SAF for this tick if operational.
            4) Market price is updated via merit-order clearing.
            5) A forward-looking consumer price forecast is regenerated and sent to investors.
            6) Agents evaluate() performance/NPV given updated context.
            7) Agents invest() and potentially add a new production site.
            8) A prospective investor is introduced and may enter if they invest.
            9) Scheduler advances; tick counters are synchronized for agents.
           10) Model and agent metrics are collected.
 
        Side Effects:
            - Updates self.states_available_feedstock, self.market_price, self.marginal_details.
            - Updates self.consumer_price_forecast and investor expectations.
            - May add new investors and/or sites to the scheduler and model state.
            - Appends records to the DataCollector.
        """
        # CLAUDE START - Clear contracts from previous year
        self.new_contracts_this_year = []
        # CLAUDE END - Clear contracts from previous year

 
        # CLAUDE FIX - Sample current supply FIRST before calculating available feedstock
        # This ensures available_feedstock and current_supply have the same baseline
        for aggregator in self.aggregators.values():
            current, _ = aggregator.sample_current_supply()
            aggregator.current_supply = current

        # Now calculate available feedstock FROM current_supply (not max_supply!)
        self.states_available_feedstock = {
            state_id: aggregator.current_supply  # ← Use CURRENT supply, not max!
            for state_id, aggregator in self.aggregators.items()
        }

        # Only operational plants pledge feedstock
        for site in self.production_sites:
            if site.operational_year <= self.schedule.time:
                pledged_feedstock = site.max_capacity * site.design_load_factor
                self.states_available_feedstock[site.state_id] = max(
                    0, self.states_available_feedstock[site.state_id] - pledged_feedstock
                )

        for aggregator in self.aggregators.values():
            aggregator.available_feedstock = self.states_available_feedstock[
                aggregator.state_id
            ]

        # Calculate contracted vs spot demand (must happen before update_supply)
        for aggregator in self.aggregators.values():
            aggregator.total_contracted_demand = 0.0
            aggregator.total_spot_demand = 0.0

        from src.utils import year_for_tick
        current_year = year_for_tick(int(self.config["start_year"]), int(self.schedule.time))

        for site in self.production_sites:
            if site.operational_year <= self.schedule.time:
                aggregator = site.aggregator
                contracted_cap = site.get_contracted_capacity(current_year)
                spot_cap = site.get_spot_capacity(current_year)

                contracted_demand = contracted_cap * site.design_load_factor * site.streamday_percentage
                spot_demand = spot_cap * site.design_load_factor * site.streamday_percentage

                aggregator.total_contracted_demand += contracted_demand
                aggregator.total_spot_demand += spot_demand

        for agent in self.schedule.agents:
            agent.update_supply()

        # CLAUDE START - Contract Renewal: Automatically renew expired contracts BEFORE production
        # IMPORTANT: This must happen BEFORE produce() to avoid production gap in renewal year
        # Check all operational sites and renew contracts that have expired
        current_year = year_for_tick(self.config["start_year"], self.schedule.time)

        for site in self.production_sites:
            # Only check sites that are operational
            if site.operational_year <= self.schedule.time:
                # Check if site has a contract that has expired
                if site.active_contract and not site.active_contract.is_active(current_year):
                    # Find the investor that owns this site
                    investor = None
                    for inv in self.investors:
                        if inv.investor_id == site.investor_id:
                            investor = inv
                            break

                    if investor:
                        # Get the aggregator for this site's state
                        aggregator = self.aggregators[site.state_id]

                        # Create a new contract to replace the expired one
                        logger.info(
                            f"Contract expired for {site.site_id}. "
                            f"Creating renewal contract in year {current_year}"
                        )

                        # Use renewal method to maintain same tier pricing
                        old_contract = site.active_contract
                        renewal_price = aggregator.renew_contract_at_same_tier(
                            existing_contract=old_contract,
                            current_year=current_year
                        )

                        # Create new contract with same terms (NO tier escalation)
                        new_contract = FeedstockContract(
                            contract_id=f"contract_{site.site_id}_renewal_{current_year}",
                            investor_id=site.investor_id,
                            aggregator_id=site.state_id,
                            plant_id=site.site_id,
                            initial_contract_price=renewal_price,  # Same tier price
                            start_year=current_year,
                            end_year=current_year + old_contract.duration,
                            duration=old_contract.duration,
                            annual_capacity=old_contract.annual_capacity,
                            contract_percentage=old_contract.contract_percentage,
                            escalation_rate=0.0,  # No escalation in tier system
                            status="active"
                        )

                        # Add to investor's contracts
                        investor.contracts.append(new_contract)

                        # Assign the new contract to the site
                        site.active_contract = new_contract

                        # Register contract with aggregator
                        aggregator.register_contract(new_contract)

                        # Track for spot price calculation
                        self.new_contracts_this_year.append(new_contract)

                        logger.info(
                            f"✓ Renewed contract for {site.site_id}: "
                            f"{new_contract.contract_percentage:.1%} coverage at "
                            f"${new_contract.initial_contract_price:.2f}/tonne"
                        )
                    else:
                        logger.warning(
                            f"Could not find investor {site.investor_id} "
                            f"to renew contract for {site.site_id}"
                        )
        # CLAUDE END - Contract Renewal: Automatically renew expired contracts BEFORE production

        # CLAUDE START - TAKE-OR-PAY: Allocate demand before production
        # When market has oversupply, we must allocate limited demand across sites.
        # This prevents overproduction and triggers take-or-pay penalties for curtailed contracts.
        current_year = year_for_tick(self.config["start_year"], self.schedule.time)
        total_demand = get_saf_demand_forecast(
            current_year,
            self.config,
            self.atf_demand_forecast
        )

        # Check if feature is enabled
        if self.config.get('enable_demand_allocation', True):
            self.demand_allocation = self.allocate_demand_to_sites(
                self.production_sites,
                total_demand,
                current_year
            )
        else:
            # Feature disabled, no allocation (everyone produces freely)
            self.demand_allocation = None
        # CLAUDE END - TAKE-OR-PAY: Allocate demand before production

        for agent in self.schedule.agents:
            agent.produce()

        self.update_consumer_price()
        updated_forecast = self.generate_price_forecast()
        self.consumer_price_forecast = updated_forecast

        logger.debug(f"Price forecast: {updated_forecast}")

        for investor in self.investors:
            investor.consumer_price_forecast = updated_forecast
            investor.current_tick = self.schedule.time

        # CLAUDE START - Calculate spot prices for each state
        current_year = year_for_tick(self.config["start_year"], self.schedule.time)
        for state_id, aggregator in self.aggregators.items():
            self.state_spot_prices[state_id] = calculate_state_spot_price(
                state_id=state_id,
                new_contracts_this_year=self.new_contracts_this_year,
                previous_spot_price=self.state_spot_prices.get(state_id),
                aggregator=aggregator
            )
            logger.debug(
                f"State {state_id} spot price: "
                f"${self.state_spot_prices[state_id]:.2f}/tonne"
            )
        # CLAUDE END - Calculate spot prices for each state

        for agent in self.schedule.agents:
            agent.evaluate()

        for agent in self.schedule.agents:
            agent.invest()
            # Note: Feedstock is updated WITHIN Investor.investment_mechanism() at line 689
            # This creates sequential "first-come, first-served" dynamics automatically
 
        logging.info("Introducing a new investor...")
        self.new_investor()
 
        self.schedule.step()
 
        # Collect data at each step
        for agent in self.schedule.agents:
            # Keep both for backward compatibility in logs/reporters
            agent.current_tick = int(self.schedule.time)
            agent.tick = int(self.schedule.time)
 
        self.datacollector.collect(self)
 
        # logging.info(f"Market price: {self.market_price}")
 
    def update_consumer_price(self) -> None:
        """
        Compute and set the current market (consumer) price based on merit-order clearing.
 
        Logic:
          - Determine applicable demand for current tick.
          - Filter sites (respecting construction/operational timelines) to create a list of currently operational sites.
          - Compute the clearing price and marginal distribution via calculate_consumer_price.
        """
 
        start_year = int(self.config["start_year"])
        current_tick = int(self.schedule.time)
        current_year = year_for_tick(start_year, current_tick)
 
       
        demand_this_tick = get_saf_demand_forecast(
            current_year, self.config, self.atf_demand_forecast
        )
        self.demand = demand_this_tick
        # CLAUDE START - Phase 2 DIFFERENTIAL ESCALATION: Pass model for year-based SRMC calculation
        operational_sites_data = find_operational_sites(
            self.production_sites, current_tick, model=self
        )
        # CLAUDE END - Phase 2 DIFFERENTIAL ESCALATION: Pass model for year-based SRMC calculation

        # CLAUDE START - Phase 2 DIFFERENTIAL ESCALATION: Escalate ATF+ price with inflation
        # ATF+ price (fossil fuel alternative) must escalate with inflation to remain economically consistent.
        # Without escalation, SAF becomes uncompetitive after ~18 years when escalated SRMC exceeds
        # the fixed €2000 cap, causing all production to halt.
        base_atf_plus_price = float(self.config["atf_plus_price"])
        inflation_rate = float(self.config.get("inflation_rate", 0.03))
        years_elapsed = current_year - start_year
        escalated_atf_plus_price = base_atf_plus_price * ((1 + inflation_rate) ** years_elapsed)
        # CLAUDE END - Phase 2 DIFFERENTIAL ESCALATION: Escalate ATF+ price with inflation

        self.market_price, self.marginal_details = calculate_consumer_price(
            operational_sites_data,
            demand_this_tick=demand_this_tick,
            atf_plus_price=escalated_atf_plus_price,  # CLAUDE: Use escalated price
        )
        for investor in self.investors:
            investor.consumer_price_forecast = [
                self.market_price,
                self.marginal_details,
            ]
 
    def generate_price_forecast(self) -> list:
        """
        Generate forward-looking consumer price expectations over the investment horizon.
 
        Uses current and expected future operational capacity, demand trajectory, and
        the ATF-plus reference price to produce a per-tick forecast starting from
        the current model tick.
 
        Returns:
            list: Projected consumer prices per future tick over the configured
            Investment_horizon_length. Index 0 corresponds to current_tick + 1.
        """
        return forecast_consumer_prices(
            model=self,
            production_sites=self.production_sites,
            demand_forecast=self.atf_demand_forecast,
            investment_horizon=int(self.config["Investment_horizon_length"]),
            atf_plus_price=self.config["atf_plus_price"],
            current_tick=self.schedule.time,
        )
 
    def allocate_demand_to_sites(
        self,
        production_sites: list,
        total_demand: float,
        current_year: int
    ) -> dict:
        """
        Allocate limited demand to production sites with contract priority.

        When total potential supply > demand, we must decide which sites produce.
        Priority order:
        1. Higher contract coverage percentage (contracted_capacity / max_capacity)
        2. Lower SRMC (cheaper production)

        This implements a realistic market clearing where:
        - Contracted capacity gets priority (firm commitments)
        - Merit order (cost) determines spot allocation
        - Sites that can't produce due to oversupply may pay take-or-pay penalties

        Parameters:
            production_sites: List of SAFProductionSite objects
            total_demand: Total market demand for this tick
            current_year: Current calendar year

        Returns:
            Dict[site_id, allocated_production]: How much each site is allowed to produce
        """
        # Calculate potential production for each operational site
        sites_data = []
        for site in production_sites:
            current_tick = self.schedule.time

            # Skip non-operational sites
            if site.operational_year > current_tick:
                continue

            contracted_cap = site.get_contracted_capacity(current_year)
            spot_cap = site.get_spot_capacity(current_year)
            contract_percentage = contracted_cap / site.max_capacity if site.max_capacity > 0 else 0

            # Potential production (what they WANT to produce)
            potential_production = site.calculate_production_output()

            sites_data.append({
                'site': site,
                'site_id': site.site_id,
                'contracted_capacity': contracted_cap,
                'spot_capacity': spot_cap,
                'contract_percentage': contract_percentage,
                'srmc': site.calculate_srmc(current_year),
                'potential_production': potential_production
            })

        # Calculate total potential supply
        total_potential_supply = sum(s['potential_production'] for s in sites_data)

        # Check if allocation is needed
        if total_potential_supply <= total_demand:
            # No oversupply, everyone produces what they want
            logger.debug(
                f"No demand allocation needed: supply={total_potential_supply:,.0f}, "
                f"demand={total_demand:,.0f}"
            )
            return {s['site_id']: s['potential_production'] for s in sites_data}

        # OVERSUPPLY: Allocate demand with priority
        logger.info(
            f"OVERSUPPLY DETECTED: supply={total_potential_supply:,.0f}, "
            f"demand={total_demand:,.0f}, excess={total_potential_supply - total_demand:,.0f}"
        )

        # Sort by priority: contract % DESC, then SRMC ASC
        sites_data.sort(key=lambda x: (-x['contract_percentage'], x['srmc']))

        # Allocate demand sequentially
        remaining_demand = total_demand
        allocation = {}

        for site_data in sites_data:
            if remaining_demand <= 0:
                # No more demand, this site produces ZERO
                allocation[site_data['site_id']] = 0.0
                logger.info(
                    f"Site {site_data['site_id']} curtailed: no demand remaining "
                    f"(potential={site_data['potential_production']:,.0f})"
                )
            else:
                # Allocate up to potential, limited by remaining demand
                allocated = min(site_data['potential_production'], remaining_demand)
                allocation[site_data['site_id']] = allocated
                remaining_demand -= allocated

                if allocated < site_data['potential_production']:
                    logger.info(
                        f"Site {site_data['site_id']} partially curtailed: "
                        f"allocated={allocated:,.0f}, potential={site_data['potential_production']:,.0f}"
                    )

        return allocation

    def new_investor(self) -> None:
        """
        Introduce and evaluate a prospective new investor entrant.

        Process:
          - Instantiate 'potential' investor with current market context.
          - Run investment_mechanism (site evaluation & potential investment).
          - Add investor to model only if asset is acquired.
        """
        current_tick = self.schedule.time
 
        new_investor = Investor(
            unique_id=Investor.generate_investor_id(),
            model=self,
            current_tick=current_tick,
            states_data=self.states_data,
        )
 
        new_investor.consumer_price_forecast = self.consumer_price_forecast
 
        new_investor.investment_mechanism(
            states_available_feedstock=self.states_available_feedstock,
            aggregators=self.aggregators,
            current_tick=current_tick,
        )
        if len(new_investor.owned_assets) > 0:
            self.investors.append(new_investor)
            self.schedule.add(new_investor)
            logging.info(f"New investor {new_investor.investor_id} with optimism factor: {new_investor.optimism_factor} added to model.")
        else:
            logging.info("New investor evaluated but did not invest.")
 
    def export_logs(self, output_dir="logs"):
        """
        Export collected simulation metrics to CSV files.
 
        Files:
          - model_log.csv: All model-level variables per tick.
          - market_metrics_log.csv: Subset (Tick, Consumer_Price, Demand, Total_Capacity) if available.
          - agent_log.csv: Full agent-level panel data.
          - <agent_type>_log.csv: One file per agent type.
        """
 
        os.makedirs(output_dir, exist_ok=True)
        model_df = self.datacollector.get_model_vars_dataframe()
        try:
            model_df = model_df.reset_index()
        except Exception:
            pass
        model_log_path = os.path.join(output_dir, "model_log.csv")
        model_df.to_csv(model_log_path, index=False)
 
        metrics_cols = [
            c
            for c in ["Tick", "Year", "Consumer_Price", "Demand", "Total_Capacity"]
            if c in model_df.columns
        ]
        if metrics_cols:
            metrics_df = model_df[metrics_cols].copy()
            metrics_df.to_csv(
                os.path.join(output_dir, "market_metrics_log.csv"), index=False
            )
 
        agent_df = self.datacollector.get_agent_vars_dataframe()
        agent_df = agent_df.reset_index()
        agent_log_path = os.path.join(output_dir, "agent_log.csv")
        agent_df.to_csv(agent_log_path, index=False)
 
        if "Type" in agent_df.columns:
            for agent_type in agent_df["Type"].unique():
                filtered_df = agent_df[agent_df["Type"] == agent_type]
                filename = f"{agent_type.lower()}_log.csv"
                filtered_df.to_csv(os.path.join(output_dir, filename), index=False)
 
        print(f"Logs exported to {output_dir}")
 
        return {
            "model_log": model_df,
            "market_metrics_log": metrics_df,
            "agent_log": agent_df,
            "by_agent_type": {agent_type: agent_df[agent_df["Type"] == agent_type] for agent_type in agent_df["Type"].unique()}
        }
 
