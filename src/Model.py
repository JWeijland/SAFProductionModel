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
                "Total_Supply": lambda m: sum(
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
                    getattr(a, "tick_production_output", None)
                    if hasattr(a, "tick_production_output")
                    else None
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
 
            for site in self.production_sites:
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
                site.operational_year = self.schedule.time
            site.produce()
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

        # CLAUDE START - Contract tracking attributes for Phase 1 implementation
        self.all_contracts = []  # Global list of all contracts
        self.state_spot_prices = {}  # Dict mapping state_id -> spot price
        self.new_contracts_this_year = []  # Contracts signed this tick
        # CLAUDE END - Contract tracking attributes for Phase 1 implementation

        # CLAUDE START - Create contracts for initial sites
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

            logging.info(
                f"Initial contract {contract.contract_id}: "
                f"{contract.contract_percentage:.1%} @ ${contract.initial_contract_price:.2f}/tonne"
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

 
        self.states_available_feedstock = {
            state_id: aggregator.max_supply
            for state_id, aggregator in self.aggregators.items()
        }
 
        for site in self.production_sites:
            pledged_feedstock = site.max_capacity * site.design_load_factor
            self.states_available_feedstock[site.state_id] = max(
                0, self.states_available_feedstock[site.state_id] - pledged_feedstock
            )
 
        for aggregator in self.aggregators.values():
            aggregator.available_feedstock = self.states_available_feedstock[
                aggregator.state_id
            ]
 
        for agent in self.schedule.agents:
            agent.update_supply()
 
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
        operational_sites_data = find_operational_sites(
            self.production_sites, current_tick
        )
 
        self.market_price, self.marginal_details = calculate_consumer_price(
            operational_sites_data,
            demand_this_tick=demand_this_tick,
            atf_plus_price=self.config["atf_plus_price"],
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
          - market_metrics_log.csv: Subset (Tick, Consumer_Price, Demand, Total_Supply) if available.
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
            for c in ["Tick", "Year", "Consumer_Price", "Demand", "Total_Supply"]
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
 
