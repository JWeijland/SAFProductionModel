import random

import numpy as np

from collections import deque

from src.Agents.SAF_Production_Site import SAFProductionSite
from src.Agents.FeedstockContract import FeedstockContract
from src.utils import year_for_tick

from typing import List, Dict, Optional, Tuple, Deque, Any

from mesa import Agent

import logging

import math

import json

 

logger = logging.getLogger("Investor")

 

class Investor(Agent):

    """

    Investor agent that:

      - Calculates EBIT and ROACE from owned assets.

      - Dynamically adjusts discount rate based on performance.

      - Evaluates potential SAF production investments (NPV-based).

      - Decides on new investments given available feedstock and state economics.

 

    Attributes:

        investor_id: Unique identifier.

        discount_rate: Current discount rate used in NPV.

        optimism_factor: Revenue optimism bias applied to forecast prices.

        owned_assets: List of SAF Production Sites owned by Investor.

        current_tick: Simulation tick.

        inv_ebit_history: EBIT across assets for discount-rate logic.

        capex_schedule: Standard CAPEX payment schedule for new sites.

        min_NPV_threshold: Minimum acceptable NPV threshold.

        investment_horizon: Investment evaluation horizon.

        total_capital_invested: Total invested CAPEX over time.

        roace_history: Previous 3 ticks' ROACE values for discount-rate adjustment.

        states_data: Per-state feedstock availability and price.

 

    DR parameters (from model.config):

    - ideal_roace: ROACE level considered “on target” when outside stability band.

    - DR_min / DR_max: Hard bounds applied after adjustment.

    - DR_sensitivity_parameter: Scales the DR change magnitude.

    - DR_target: DR to move toward when ROACE is within the stability band.

    - min_ROACE_stability / max_ROACE_stability: Band within which DR moves toward DR_target.

 

    """

 

    inv_counter = 0

 

    def __init__(

        self,

        unique_id: Any,

        model: Any,

        current_tick: int,

        states_data: Dict[str, Dict[str, float]],

        discount_rate: Optional[float] = None,

        optimism_factor: Optional[float] = None,

        owned_assets: Optional[List[Tuple[Dict, SAFProductionSite]]] = None,

        total_capital_invested: float = 0.0,

    ) -> None:

        """

        Initialise an Investor.

 

        Logic:

          - Sample discount_rate / optimism_factor if not provided.

          - Build baseline CAPEX schedule (adjustable for dynamic learning curve).

 

        Parameters:

            unique_id: Unique identifier.

            model: Mesa model reference.

            current_tick: Current simulation tick.

            states_data: Mapping of state_id -> feedstock info.

            discount_rate: Optional override for discount rate.

            optimism_factor: Optional override for price optimism factor.

            owned_assets: Optional pre-existing assets.

            total_capital_invested: Starting invested capital.

        """

        super().__init__(unique_id, model)

        self.investor_id = unique_id

        self.discount_rate: float = (

            discount_rate

            if discount_rate

            else random.uniform(

                float(model.config["DR_sample_min"]),

                float(model.config["DR_sample_max"]),

            )

        )

 

        self.optimism_factor: float = (

            optimism_factor

            if optimism_factor

            else random.uniform(

                model.config["Optimism_factor_sample_min"],

                model.config["Optimism_factor_sample_max"],

            )

        )

        self.owned_assets = owned_assets if owned_assets is not None else []

        self.num_owned_assets: int = 0

        self.current_tick: int = current_tick

        self.inv_ebit_history = deque(maxlen=3)

        total_cost = (

            int(model.config["capex_total_cost"])

            if not model.booleans["capex_decrease"]

            else self.get_dynamic_capex(

                int(model.config["capex_total_cost"]),

                self.current_tick,

                model.config["capex_annual_decrease"],

            )

        )

        construction_time = int(model.config["saf_plant_construction_time"])

        self.capex_schedule = [total_cost / construction_time] * construction_time

        self.min_NPV_threshold: float = model.config["min_NPV_threshold"]

        self.investment_horizon: int = int(model.config["Investment_horizon_length"])

        self.total_capital_invested = (

            total_capital_invested if total_capital_invested else 0.0

        )

        self.roace_history: Deque[float] = deque(maxlen=3)

        self.states_data: Dict[str, Dict[str, float]] = states_data

        self.ideal_roace: float = model.config["ideal_roace"]

        self.min_dr: float = model.config["DR_min"]

        self.max_dr: float = model.config["DR_max"]
        self.contracts: List[FeedstockContract] = []



    def adjust_discount_rate(

        self,

    ) -> None:

        """

        Adjust discount rate based on trailing ROACE performance.

 

        Equation:

          - ROACE = EBIT / Total Capital Invested

 

        ROACE is then averaged over past 3 years' performance.

 

        Logic:

          - Compute investor's total EBIT over all it's owned assets.

          - Compute ROACE from averaged EBIT and total capital invested.

          - If avg ROACE is within [min_ROACE_stability, max_ROACE_stability]:

              Move DR toward DR_target by delta = (DR_target - current_DR) * DR_sensitivity_parameter.

            Else:

              Move DR opposite the ROACE deviation from ideal_roace:

            delta = -(avg_roace - ideal_roace) * DR_sensitivity_parameter.

        - Apply: discount_rate = clamp(current_DR + delta, DR_min, DR_max).

        """

 

        logger.info(

            f"Investor_ID: {self.investor_id} evaluating discount rate: {self.discount_rate:.3f}"

        )

 

        if not self.owned_assets:

            return

 

        total_ebit = 0

        raw_total_ebit = 0

 

        DR_sensitivity_parameter = self.model.config["DR_sensitivity_parameter"]

        DR_target = self.model.config["DR_target"]

 

        for asset, _ in self.owned_assets:

            total_ebit += asset["ebit_history"][-1]

            raw_total_ebit += asset["raw_ebit_history"][-1]

 

        if math.isclose(self.total_capital_invested, 0.0):

            return

 

        self.ebit = total_ebit

        self.raw_ebit = raw_total_ebit

 

        roace = total_ebit / self.total_capital_invested

        raw_roace = raw_total_ebit / self.total_capital_invested

        self.roace_history.append(roace)

        self.roace = roace

        self.raw_roace = raw_roace

 

        avg_roace = sum(self.roace_history) / len(self.roace_history)

 

        min_roace = self.model.config["ROACE_stability_min"]

        max_roace = self.model.config["ROACE_stability_max"]

 

        if min_roace <= avg_roace <= max_roace:

            if self.discount_rate < DR_target:

                delta_dr = (self.discount_rate - DR_target) * DR_sensitivity_parameter

            elif self.discount_rate > DR_target:

                delta_dr = -(self.discount_rate - DR_target) * DR_sensitivity_parameter

            else:

                delta_dr = 0

 

        else:

            delta_dr = -(avg_roace - self.ideal_roace) * DR_sensitivity_parameter

 

        self.discount_rate = self.discount_rate + delta_dr

 

        self.discount_rate = max(self.min_dr, min(self.discount_rate, self.max_dr))

 

        logger.info(

            f"Investor has average ROACE of {avg_roace:.3f}, adjusting DR to {self.discount_rate:.3f} accordingly"

        )

 

    def evaluate_investment(

        self, site: SAFProductionSite, current_tick: int, operational: bool = False

    ) -> Dict:

        """

        Evaluate a potential SAF site by calculating NPV.

 

        Equation:

          - NPV = Σ (Revenue_t - Cost_t) / (1 + discount_rate)^t  for t in 1 to investment_horizon

 

        Assumptions:

          - Uses design load factor determined by available feedstock in specific state.

          - Excludes profit margin from SRMC for pure economic NPV.

 

        Parameters:

            site: Site under evaluation.

            current_tick: Current simulation tick.

            operational: If True, site considered already operational (affects metadata only).

 

        Returns:

            Asset metadata dict (site_id, state_id, npv, capex_schedule, tick_built, ebit_history).

        """

        production_output: float = site.max_capacity * site.design_load_factor

        srmc_no_profit: float = site.srmc - site.profit_margin

        capex_schedule: List = self.capex_schedule

        model_start_year = int(self.model.config["start_year"])
        current_year_for_npv = year_for_tick(model_start_year, current_tick)
        contract_price = site.aggregator.get_current_market_price(current_year_for_npv)

        npv: float = self.calculate_npv(
            site, production_output, srmc_no_profit, capex_schedule,
            contract_price=contract_price,
            start_year=current_year_for_npv
        )

        asset = {

            "site_id": site.site_id,

            "state_id": site.state_id,

            "npv": npv,

            "capex_schedule": capex_schedule,

            "tick_built": current_tick,

            "ebit_history": deque(maxlen=3),

            "raw_ebit_history": deque(maxlen=3),

        }

 

        return asset

 

    def investment_mechanism(

        self,

        states_available_feedstock: Dict[str, float],

        aggregators: Dict[str, object],

        current_tick: int,

    ) -> None:

        """

        Scan all states and attempt to invest in the highest NPV site above threshold (min_NPV_threshold).

 

        Tie-breaking:

          - Randomly selects among equal-top NPV candidates above threshold.

 

        Side-effects:

          - Appends new site to model if investment occurs.

          - Records evaluation diagnostics for logging & analysis.

 

        Parameters:

           states_available_feedstock: Remaining feedstock by state after allocations.

           aggregators: State-level feedstock aggregator- one per state.

           current_tick: Current simulation tick.

        """

 

        self.investment_evaluations = []

        self.last_evaluation_summary = None

        self.last_investment_made = False

        self.invested_site_id = None

        self.invested_site_state = None

        self.invested_site_npv = None

 

        candidate_assets = []

        highest_npv = None

 

        logger.info(

            f"Investor_ID: {self.investor_id} with discount rate: {self.discount_rate} and optimism factor: {self.optimism_factor} evaluating sites"

        )

 

        for state_id, info in self.states_data.items():

            feedstock_price = info["feedstock_price"]

            available_feedstock = states_available_feedstock[state_id]

 

            design_load_factor = min(

                1, available_feedstock / self.model.config["max_capacity"]

            )

            aggregator = aggregators[state_id]

 

            site = SAFProductionSite(

                unique_id=SAFProductionSite.generate_site_id(state_id),

                model=self.model,

                state_id=state_id,

                investor_id=self.investor_id,

                config=self.model.config,

                design_load_factor=design_load_factor,

                aggregator=aggregator,

                capex_schedule=self.capex_schedule,

            )

            site.produce()

            asset = self.evaluate_investment(site, current_tick)

 

            self.investment_evaluations.append(

                {

                    "state_id": state_id,

                    "feedstock_price": feedstock_price,

                    "available_feedstock": available_feedstock,

                    "design_load_factor": round(design_load_factor, 4),

                    "npv": (

                        float(asset["npv"]) if asset.get("npv") is not None else None

                    ),

                }

            )

 

            if asset["npv"] > self.min_NPV_threshold:

                if highest_npv is None or asset["npv"] > highest_npv:

                    highest_npv = asset["npv"]

                    candidate_assets = [(asset, site)]

                elif math.isclose(asset["npv"], highest_npv):

                    candidate_assets.append((asset, site))

 

            logger.debug(

                f"Evaluating state: {state_id}  Feedstock price: {feedstock_price}, Available feedstock: {available_feedstock}  Design Load Factor: {design_load_factor:.2f}"

            )

            logger.debug(f"  Calculated NPV: {asset['npv']:.2f}")

 

        try:

            self.last_evaluation_summary = json.dumps(self.investment_evaluations)

        except Exception:

            self.last_evaluation_summary = str(self.investment_evaluations)

 

        if candidate_assets:

            best_asset, best_site = random.choice(candidate_assets)

            logger.info(

                f"Investing in site {best_asset['site_id']} in state {best_asset['state_id']} with NPV {best_asset['npv']:.2f}"

            )

            print("Investment occurred")

            self.invested_site_id = best_asset["site_id"]

            self.invested_site_state = best_asset["state_id"]

            self.invested_site_npv = float(best_asset["npv"])

            self.owned_assets.append((best_asset, best_site))

            self.num_owned_assets += 1

            self.model.production_sites.append(best_site)

            self.model.schedule.add(best_site)

            self.model.states_available_feedstock[best_asset['state_id']] -= best_site.max_capacity * best_site.design_load_factor

            aggregator = aggregators[best_asset['state_id']]
            current_year = year_for_tick(
                int(self.model.config["start_year"]),
                current_tick
            )

            contract = self.create_contract(
                aggregator=aggregator,
                plant=best_site,
                current_year=current_year
            )

            self.model.all_contracts.append(contract)
            self.model.new_contracts_this_year.append(contract)
            aggregator.register_contract(contract)
            best_site.active_contract = contract

            logger.info(
                f"Created contract {contract.contract_id} with "
                f"{contract.contract_percentage:.1%} coverage at "
                f"${contract.initial_contract_price:.2f}/tonne"
            )

        else:

            logger.info("No investment made this year.")

 

    def calculate_ebit(

        self,

        site: SAFProductionSite,

        market_price: float,

        capex: float,

        annual_load_factor: float,
        current_year: int = None,

    ) -> float:

        """
        Compute EBIT for a site in current year.



        Equation:

          - EBIT = Revenue - Cost



        Rules:

          - If plant has active contract: MUST produce contracted volume (80-90% capacity)
            even if effective SRMC > market price (forced loss from contract obligation)

          - If no contract and SRMC > market price: site does not run; charges idle opex

          - If site is marginal => output scaled by marginal_multiplier

          - Otherwise full available production



        Parameters:

            site: Production site object.

            market_price: Clearing SAF market price.

            capex: CAPEX draw this tick (if still constructing).

            annual_load_factor: Realised annual load factor from aggregator (factors in feedstock variability).

            current_year: Current simulation year (for contract-aware SRMC calculation).



        Returns:

            EBIT (float) - Can be negative if forced to produce under contract at loss.

        """

        # Use blended feedstock costs (contract + spot) instead of fixed aggregator price
        if current_year is not None:
            # Calculate escalation factor for non-feedstock costs
            start_year = int(self.model.config["start_year"])
            years_elapsed = current_year - start_year
            market_escalation_rate = float(self.model.config.get("market_escalation_rate", 0.02))
            escalation_factor = (1 + market_escalation_rate) ** years_elapsed

            # Get spot price for this state
            spot_price = self.model.state_spot_prices.get(
                site.state_id,
                site.aggregator.feedstock_price  # Fallback to aggregator price
            )

            # Get blended feedstock cost (contract + spot)
            blended_feedstock_cost = self.get_feedstock_cost(
                plant=site,
                current_year=current_year,
                spot_price=spot_price
            )

            # ALL cost components escalate at market rate (consistency with calculate_srmc)
            opex_escalated = site.opex * escalation_factor
            transport_escalated = site.transport_cost * escalation_factor
            margin_escalated = site.profit_margin * escalation_factor

            # Calculate contract-aware SRMC with escalated costs
            effective_srmc = (
                blended_feedstock_cost
                + opex_escalated
                + transport_escalated
                + margin_escalated
            )
        else:
            # Backwards compatibility: use original SRMC if no current_year provided
            effective_srmc = site.srmc
            opex_escalated = site.opex
            transport_escalated = site.transport_cost
            margin_escalated = site.profit_margin



        full_production_volume = site.year_production_output

        # Find if this plant has an active contract that enforces minimum production
        active_contract = self.get_contract_for_plant(site.site_id, current_year)


        if effective_srmc > market_price:
            if active_contract is not None:
                # CONTRACT OBLIGATION: Must produce contracted volume even at loss!
                contracted_volume = active_contract.contracted_volume

                # Calculate production considering annual load factor variability
                # Contracted volume is based on design capacity, scale by annual_load_factor
                min_production = min(contracted_volume * annual_load_factor, full_production_volume)

                # Calculate FORCED production at loss
                srmc_no_profit = effective_srmc - margin_escalated
                revenue = market_price * min_production
                variable_cost = srmc_no_profit * min_production

                # Idle opex on unused capacity
                idle_capacity = full_production_volume - min_production
                idle_cost = opex_escalated * idle_capacity

                total_cost = variable_cost + idle_cost + capex

                site.tick_production_output = min_production
                site.ebit = revenue - total_cost

                logger.warning(
                    f"Site {site.site_id} FORCED to produce {min_production:.0f} tonnes due to contract: "
                    f"Effective SRMC €{effective_srmc:.2f} > Market €{market_price:.2f} "
                    f"→ EBIT: €{site.ebit:,.0f} (LOSS from contract obligation)"
                )

                return site.ebit
            else:
                # No contract - normal behavior: don't produce if unprofitable
                cost = (
                    opex_escalated
                    * site.max_capacity
                    * site.design_load_factor
                    * annual_load_factor
                    * site.streamday_percentage
                )

                site.tick_production_output = 0.0
                site.ebit = -cost

                logger.debug(
                    f"Site {site.site_id} NOT producing (no contract): "
                    f"Effective SRMC €{effective_srmc:.2f} > Market €{market_price:.2f}"
                )

                return -cost

        srmc_no_profit = effective_srmc - margin_escalated



        if math.isclose(effective_srmc, market_price):

            marginal_multiplier = self.calculate_marginal_output(

                production_output=0, marginal_details=self.model.marginal_details

            )

            production_volume = full_production_volume * marginal_multiplier

            logger.info(

                f"Production shared at {marginal_multiplier} with output of {production_volume}"

            )

        else:

            production_volume = full_production_volume



        site.tick_production_output = production_volume



        revenue = market_price * production_volume


        cost = (srmc_no_profit * production_volume + capex) + (

            opex_escalated * (full_production_volume - production_volume)

        )

        # If site was curtailed due to oversupply, it must pay take-or-pay penalty
        # for contracted feedstock it committed to buy but couldn't use
        take_or_pay_penalty = getattr(site, 'take_or_pay_penalty', 0.0)

        if take_or_pay_penalty > 0:
            logger.info(
                f"Site {site.site_id} EBIT before penalty: ${revenue - cost:,.0f}, "
                f"penalty: ${take_or_pay_penalty:,.0f}"
            )

        site.ebit = revenue - cost - take_or_pay_penalty

        return site.ebit

 

    def annual_update(

        self,

        market_price: float,

        current_tick: int,

    ) -> None:

        """

        Annual update of financial metrics across owned assets.

 

        Actions:

          - Apply current CAPEX (if under construction).

          - Calculates EBIT (if under construction, assigns capex without calculation).

          - Append EBIT to per-asset history.

 

        Parameters:

            market_price: Current market price.

            current_tick: Current tick.

        """

        self.current_tick = current_tick

        for asset, site in self.owned_assets:

 

            capex = (

                asset["capex_schedule"][0]

                if current_tick < site.operational_year

                else 0

            )

 

            self.total_capital_invested += capex

            if site.operational_year <= current_tick:
                current_year = year_for_tick(
                    int(self.model.config["start_year"]),
                    current_tick
                )

                ebit = self.calculate_ebit(

                    site, market_price, capex, site.aggregator.annual_load_factor,
                    current_year=current_year

                )



                asset["ebit_history"].append(ebit)

                asset["raw_ebit_history"].append(ebit)

            else:

                site.ebit = -capex

                asset["ebit_history"].append(-capex)

                asset["raw_ebit_history"].append(0)

 

    def get_forecast_price(self, year: int) -> float:

        """

        Get forecasted SAF consumer price for a future year (bounded by last known value).

 

        Parameters:

            year: Year of forecast desired.

 

        Returns:

            Forecast price and marginal market details.

        """

 

        forecast = self.consumer_price_forecast

 

        prices = [item[0] for item in forecast]

        details = [item[1][0] for item in forecast]

        if year < len(forecast):

            price = prices[year - 1]

            detail = details[year - 1]

        else:

            price = prices[-1]

            detail = details[-1]

        return price, detail

 

    def calculate_npv(

        self,

        site: SAFProductionSite,

        production_output: float,

        srmc_no_profit: float,

        capex_schedule: List[float],
        contract_price: float = None,
        start_year: int = None,

    ) -> float:

        """
        # NPV calculation now accounts for blended feedstock costs with contract escalation.
        # When contract_price and start_year are provided, the NPV forecast includes:
        # - 85% contract pricing with 3% annual escalation for first 20 years
        # - 15% spot pricing throughout
        # - 100% spot pricing after contract expires (year 21+)
        # This provides realistic long-term cost evolution for investment decisions.

        Calculate the net present value (NPV) of a candidate site.

 

        Equation:

          - NPV = ∑ (revenue - cost) / (1 + discount_rate)^t

 

        Cash flow model:

          - Construction years: no production, only CAPEX.

          - Operational years: production at `production_output`, unless the site is marginal (SRMC ~= forecasted price), in which case only a fraction is sold determined by `calculate_marginal_output` and the market's `marginal_details`.

          - Revenue = forecasted_price * sold_output * optimism_factor

          - Cost = srmc_no_profit * sold_output

                 + CAPEX draw in year t (if present)

                 + raw OPEX for unused capacity (production_output - sold_output)

 

        Args:

            site: The SAFProductionSite being evaluated (used for SRMC and marginal checks).

            production_output: Annual production volume at design load (post-construction).

            srmc_no_profit: SRMC excluding profit margin (per unit of output).

            capex_schedule: Yearly CAPEX draws; may be shorter than the investment horizon.

 

        Returns:

            float: Present value of the stream of (revenue - cost) across the horizon.

 

        Notes:

            - Uses `self.optimism_factor` to bias revenue.

            - Uses `self.discount_rate` for discounting.

            - Market price and marginal details are retrieved via `get_forecast_price(t)`.

        """

        npv = 0.0
        # Determine if we should use contract pricing
        use_contract_pricing = (contract_price is not None and start_year is not None)
        # Get contract percentage from config (use average of min/max range)
        contract_percentage_min = float(self.model.config.get("contract_percentage_min", 0.70))
        contract_percentage_max = float(self.model.config.get("contract_percentage_max", 0.70))
        contract_percentage = (contract_percentage_min + contract_percentage_max) / 2
        # Get escalation rates from config
        contract_escalation_rate = float(self.model.config.get("contract_escalation_rate", 0.03))  # CPI
        market_escalation_rate = float(self.model.config.get("market_escalation_rate", 0.02))  # CPI - tech
        contract_duration = int(self.model.config.get("contract_duration", 20))
        construction_time = self.model.config["saf_plant_construction_time"]

        for t in range(1, self.investment_horizon + 1):

            marginal_multiplier = 1

            forecasted_price, marginal_details = self.get_forecast_price(t)



            # When plant is still in construction

            if t <= construction_time:

                current_production_output = 0

                revenue = 0

                cost = capex_schedule[t - 1]

            else:
                # Operational phase - calculate SRMC with differential escalation
                if use_contract_pricing:
                    # Years since plant became operational
                    years_operational = t - construction_time

                    # Forecasted spot price with market escalation (2%/year with tech improvement)
                    # Start from contract_price (current market) and escalate at market rate
                    spot_forecast = contract_price * ((1 + market_escalation_rate) ** years_operational)

                    if years_operational <= contract_duration:
                        # Contract is active - use blended pricing
                        # Contract escalates at 3%/year (CPI) - FASTER than market
                        escalated_contract_price = contract_price * ((1 + contract_escalation_rate) ** years_operational)

                        # Spot escalates at 2%/year (CPI - tech) - SLOWER than contract
                        blended_feedstock_cost = (
                            escalated_contract_price * contract_percentage +
                            spot_forecast * (1 - contract_percentage)
                        )

                        # Growing gap: contract (3%) vs market (2%) creates increasing cost penalty
                    else:
                        # Contract expired - use 100% spot pricing (market escalation)
                        blended_feedstock_cost = spot_forecast

                    # All cost components escalate at market rate
                    escalation_factor_npv = (1 + market_escalation_rate) ** years_operational
                    opex_escalated_npv = site.opex * escalation_factor_npv
                    transport_escalated_npv = site.transport_cost * escalation_factor_npv
                    margin_escalated_npv = site.profit_margin * escalation_factor_npv

                    # Calculate contract-aware SRMC for this year with ALL escalated costs
                    effective_srmc_for_npv = (
                        blended_feedstock_cost +
                        opex_escalated_npv +
                        transport_escalated_npv +
                        margin_escalated_npv
                    )
                    effective_srmc_no_profit = effective_srmc_for_npv - margin_escalated_npv
                else:
                    # Backwards compatibility: use original SRMC
                    effective_srmc_for_npv = site.srmc
                    effective_srmc_no_profit = srmc_no_profit

                if effective_srmc_for_npv > forecasted_price:

                    current_production_output = 0

                elif effective_srmc_for_npv < forecasted_price:

                    current_production_output = production_output

                else:

                    current_production_output = production_output

                    # If the plant would contribute to the marginal

                    marginal_multiplier = self.calculate_marginal_output(

                        production_output, marginal_details

                    )

                    current_production_output = production_output * marginal_multiplier



                revenue = (

                    forecasted_price * current_production_output * self.optimism_factor

                )



                raw_opex = self.model.config["opex"]

                # Cost is the srmc for the production output, plus the raw opex for the remaining capacity they are not producing.

                cost = (

                    effective_srmc_no_profit * current_production_output

                    + (production_output - current_production_output) * raw_opex

                )

 

            npv_tic = (revenue - cost) / ((1 + self.discount_rate) ** t)

            npv += npv_tic

        return npv

 

    def calculate_marginal_output(

        self, production_output: float, marginal_details: Dict[str, Any]

    ) -> float:

        """

        Calculate and update the marginal output for a given site based on market conditions.

 

        Parameters:

            site: The SAFProductionSite instance to update.

            marginal_details: Dictionary containing details about the marginal market conditions.

        """

 

        supply_at_marginal = marginal_details["supply_at_marginal"]

        needed_from_marginal = marginal_details["needed_from_marginal"]

 

        new_supply_at_marginal = supply_at_marginal + production_output

 

        percentage_sold = needed_from_marginal / new_supply_at_marginal

 

        return percentage_sold

 

    @classmethod

    def generate_investor_id(cls) -> str:

        cls.inv_counter += 1

        """

        Generate a unique investor ID.

 

        Returns:

            Unique investor identifier string.

        """

        return f"inv_{cls.inv_counter:02d}"

 

    def __repr__(self) -> str:

        """Returns a string representation of the Investor instance for debugging."""

        return (

            f"Investor(id='{self.investor_id}', "

            f"discount_rate={self.discount_rate:.4f}, "

            f"optimism_factor={self.optimism_factor:.2f}, "

            f"owned_assets={len(self.owned_assets)}, "

            f"total_capital_invested={self.total_capital_invested:.2f})"

        )

 

    def get_dynamic_capex(

        self, capex_total_cost: float, year: int, capex_decrease: float

    ) -> float:

        """Apply a linear annual CAPEX reduction.

 

        Args:

            capex_total_cost: Baseline total CAPEX.

            year: Non-negative simulation year index.

            capex_decrease: Fractional decrease per year (e.g., 0.03).

 

        Returns:

            float: Adjusted CAPEX for the given year.

 

        """

        if year < 0:

            raise ValueError("Year must be a non-negative integer.")

 

        decrease = capex_decrease * year

 

        return capex_total_cost * (1 - decrease)

    def create_contract(
        self,
        aggregator,
        plant: SAFProductionSite,
        current_year: int,
    ) -> FeedstockContract:
        """
        Create a new 20-year feedstock contract for a plant.

        This is called when the investor decides to invest in a new plant.
        The contract locks in feedstock supply at the plant's current SRMC
        with 3% annual escalation.

        Parameters:
            aggregator: FeedstockAggregator for the plant's state
            plant: SAFProductionSite that needs feedstock
            current_year: Year contract is signed

        Returns:
            FeedstockContract object

        Side effects:
            - Adds contract to self.contracts
        """
        # Investor chooses coverage percentage (80-90%)
        contract_percentage = self.decide_contract_percentage()

        # Get contract duration from model config
        duration = int(self.model.config.get("contract_duration", 20))

        # TIERED PRICING - Calculate effective capacity and contract volume
        # The annual_capacity should reflect the ACTUAL maximum capacity considering:
        # - design_load_factor: Plant's design utilization
        # - contracted_load_factor: Aggregator's reliability for contracted feedstock
        # - streamday_percentage: Operating days percentage
        effective_max_capacity = (
            plant.max_capacity *
            plant.design_load_factor *
            aggregator.contracted_load_factor *
            plant.streamday_percentage
        )

        # Contract volume = percentage of effective capacity
        contract_annual_volume = effective_max_capacity * contract_percentage

        # TIERED PRICING - Allocate contract and get tier-based price (NO ESCALATION)
        # This permanently reserves capacity and returns the weighted tier price
        contract_price = aggregator.allocate_contract(contract_annual_volume, current_year)

        logger.info(
            f"Investor {self.investor_id} securing contract for plant {plant.site_id}:"
        )
        logger.info(
            f"  Effective max capacity: {effective_max_capacity:.0f} ton/year"
        )
        logger.info(
            f"  Contract coverage: {contract_percentage:.1%} = {contract_annual_volume:.0f} ton/year"
        )
        logger.info(
            f"  Contract price: ${contract_price:.2f}/ton (FIXED, tier-based, NO escalation)"
        )

        # Create contract with FIXED tier-based price (NO escalation)
        contract = FeedstockContract(
            contract_id=f"contract_{plant.site_id}",
            investor_id=self.investor_id,
            aggregator_id=aggregator.state_id,
            plant_id=plant.site_id,
            initial_contract_price=contract_price,  # Tier-based price (fixed)
            start_year=current_year,
            end_year=current_year + duration,
            annual_capacity=effective_max_capacity,  # Effective capacity
            contract_percentage=contract_percentage,
            escalation_rate=0.0,  # NO ESCALATION in tier system
            duration=duration,
            status="active"
        )

        # Add to investor's contract list
        self.contracts.append(contract)

        logger.info(
            f"Contract {contract.contract_id} created: "
            f"{contract.contracted_volume:.0f} ton/year @ ${contract_price:.2f}/ton, "
            f"duration: {duration} years (year {current_year}-{current_year+duration})"
        )

        return contract

    def decide_contract_percentage(self) -> float:
        """
        Investor chooses what percentage of plant capacity to cover with contract.

        Phase 1: Simple random uniform sampling in [0.80, 0.90].
        Phase 2+: Could be based on risk aversion, price expectations, etc.

        Returns:
            Contract percentage between 0.80 and 0.90
        """
        min_pct = float(self.model.config.get("contract_percentage_min", 0.80))
        max_pct = float(self.model.config.get("contract_percentage_max", 0.90))

        return random.uniform(min_pct, max_pct)

    def get_feedstock_cost(
        self,
        plant: SAFProductionSite,
        current_year: int,
        spot_price: float
    ) -> float:
        """
        Calculate blended feedstock cost for a plant.

        Returns weighted average of:
        - Contract price (80-90% coverage) - escalated from initial price
        - Spot price (10-20% remaining)

        If no contract exists for the plant, returns full spot price.

        Parameters:
            plant: SAFProductionSite to calculate cost for
            current_year: Current calendar year
            spot_price: Current spot market price for the plant's state

        Returns:
            Blended feedstock cost in USD/tonne

        Example:
            Contract: 85% @ $695/tonne
            Spot: 15% @ $720/tonne
            Blended: 0.85 * 695 + 0.15 * 720 = $699.75/tonne
        """
        # Find active contract for this plant
        contract = next(
            (c for c in self.contracts
             if c.plant_id == plant.site_id
             and c.is_active(current_year)),
            None
        )

        # No contract - use full spot price
        if contract is None:
            return spot_price

        # Get escalated contract price for this year
        contract_price = contract.get_price_for_year(current_year)

        # Calculate blended cost
        contract_cost = contract_price * contract.contract_percentage
        spot_cost = spot_price * (1 - contract.contract_percentage)

        blended_cost = contract_cost + spot_cost

        logger.debug(
            f"Plant {plant.site_id} year {current_year}: "
            f"{contract.contract_percentage:.1%} @ ${contract_price:.2f} + "
            f"{1-contract.contract_percentage:.1%} @ ${spot_price:.2f} = "
            f"${blended_cost:.2f}/tonne"
        )

        return blended_cost

    def get_contract_for_plant(
        self,
        plant_id: str,
        current_year: int
    ) -> Optional[FeedstockContract]:
        """
        Find active contract for a specific plant.

        Helper method to retrieve the contract (if any) for a given plant
        in a specific year. Used by calculate_ebit() to check contract obligations.

        Parameters:
            plant_id: ID of the plant to find contract for
            current_year: Year to check contract activity

        Returns:
            FeedstockContract if active contract exists, None otherwise

        Example:
            >>> contract = investor.get_contract_for_plant("site_PUNJAB_001", 2034)
            >>> if contract:
            ...     print(f"Plant has {contract.contract_percentage:.0%} coverage")
        """
        contract = next(
            (c for c in self.contracts
             if c.plant_id == plant_id
             and c.is_active(current_year)),
            None
        )
        return contract



    def update_supply(self) -> None:

        """Mesa step, investor not involved."""

        pass

 

    def produce(self) -> None:

        """Mesa step, investor not involved."""

        pass

 

    def evaluate(self) -> None:

        """

        Stage- evaluate:

         - Updates asset financials for current tick.

         - Adjusts discount rate.

        """

        self.annual_update(

            market_price=self.model.market_price,

            current_tick=self.model.schedule.time,

        )

 

        self.adjust_discount_rate()

 

        self.current_tick = self.model.schedule.time

 

    def invest(self) -> None:

        """

        Stage- invest:

         - Runs investment mechanism with current feedstock availability.

        """

 

        self.investment_mechanism(

            states_available_feedstock=self.model.states_available_feedstock,

            aggregators=self.model.aggregators,

            current_tick=self.model.schedule.time,

        )

 

 