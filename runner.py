from __future__ import annotations
from pathlib import Path
import logging
import sys
import pandas as pd
from io import StringIO
from typing import Optional, Iterable
import os
from helper import SLIDER_CONFIG_KEYS_RANGE
 
# Resolve project root
_REPO_ROOT = Path(__file__).resolve().parent
_INPUT_DIR = _REPO_ROOT / "input"
_OUTPUT_DIR = _REPO_ROOT / "output"
 
# Ensure repo root is on sys.path
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
 
from src.Model import SAFMarketModel
 
def _load_inputs_from_csv(scenario: str, feedstock_scenario: str, config_store: dict) -> tuple[dict, dict, dict, dict]:
    scenario_file = f"{scenario.lower()}_demand_values.csv"
    states_file = f"states_data_{feedstock_scenario.lower()}.csv"
 
    states_df = pd.read_csv(_INPUT_DIR / states_file)
    demand_df = pd.read_csv(_INPUT_DIR / scenario_file)
 
    states_df.columns = states_df.columns.str.strip()
    demand_df.columns = demand_df.columns.str.strip()
 
    demand = dict(zip(demand_df["year"], demand_df["value"]))
    states_data = {
        row["state"]: {
            "max_supply": row["max_supply (Tonnes/year)"],
            "feedstock_price": row["feedstock_price(USD/Tonne)"],
            "feedstock_type": row["feedstock_type"],
        }
        for _, row in states_df.iterrows()
    }
    #process config sliders into _min and _max keys if they are ranges
    if config_store:
        for key in list(config_store.keys()):
            if key in SLIDER_CONFIG_KEYS_RANGE and isinstance(config_store[key], list) and len(config_store[key]) == 2:
                config_store[f"{key}_min"] = config_store[key][0]
                config_store[f"{key}_max"] = config_store[key][1]
                del config_store[key]
 
    return states_data, demand, config_store
 
def trim_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how='all')
 
def run_market_model(
    *,
    booleans: dict,
    config: dict,
    states_data: dict,
    demand: dict,
    seed: int | None,
    steps: int,
    log_filename: str = "simulation_run.log",
) -> tuple[
    pd.DataFrame, pd.DataFrame, Path, str,
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Run the SAF market model and return model/agent data, log file path, in-memory logs,
    and agent-type-specific logs (FeedstockAggregator, SAFProductionSite, Investor).
    """
 
    # Prepare log file path
    log_path = _OUTPUT_DIR / log_filename
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
    # Set up dual logging: file + in-memory
    log_stream = StringIO()
    logger = logging.getLogger("SAFLogger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
 
    stream_handler = logging.StreamHandler(log_stream)
    file_handler = logging.FileHandler(log_path, mode="w")
 
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
 
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
 
    # Initialize and run model
    model = SAFMarketModel(
        config=config,
        states_data=states_data,
        atf_demand_forecast=demand,
        booleans=booleans,
        seed=seed
    )
 
    for step in range(steps):
        logger.info(f"\n--- Step {step} ---")
        model.step()
        logger.info(f"Market price: {getattr(model, 'market_price', 'n/a')}")
 
    # Export logs to disk and capture them in memory
    logs = model.export_logs()
 
    model_df = trim_empty_columns(logs["model_log"])
    market_metric_log = trim_empty_columns(logs["market_metrics_log"])
    # agent_log = logs["agent_log"]
 
    feedstock_aggregator_log = trim_empty_columns(logs["by_agent_type"].get("FeedstockAggregator"))
    investor_log = trim_empty_columns(logs["by_agent_type"].get("Investor"))
    saf_production_site_log = trim_empty_columns(logs["by_agent_type"].get("SAFProductionSite"))
 
    # Collect results
    model_vars = model.datacollector.get_model_vars_dataframe()
    agent_vars = model.datacollector.get_agent_vars_dataframe()
 
    model_vars.to_csv(_OUTPUT_DIR / "model_vars.csv", index=False)
    agent_vars.to_csv(_OUTPUT_DIR / "agent_vars.csv", index=False)
    feedstock_aggregator_log.to_csv(_OUTPUT_DIR / "feedstock_aggregator_log.csv", index=False)
    saf_production_site_log.to_csv(_OUTPUT_DIR / "saf_production_site_log.csv", index=False)
    investor_log.to_csv(_OUTPUT_DIR / "investor_log.csv", index=False)
    market_metric_log.to_csv(_OUTPUT_DIR / "market_metric_log.csv", index=False)
 
    return (
        model_df,
        feedstock_aggregator_log,
        saf_production_site_log,
        investor_log,
        market_metric_log,
    )


def run_market_model_csv(*, scenario: str, feedstock_scenario: str, steps: int, log_filename: str = "simulation_run.log", config_store: dict = None, boolean_config_store: dict = None, seed = None):
    states_data, demand, config_store = _load_inputs_from_csv(scenario, feedstock_scenario, config_store)


    return run_market_model(
        booleans=boolean_config_store,
        config=config_store,
        states_data=states_data,
        demand=demand,
        steps=steps,
        log_filename=log_filename,
        seed=seed,
    )
 
def run_market_model_csv_batch(
    *, scenario: str, feedstock_scenario: str, steps: int,
    config_store: dict = None, boolean_config_store: dict = None, base_seed=None, runs=None, prefix=None
):
    import pandas as pd
 
    keys_to_keep = {
        "model": ["Year", "Consumer_Price","Market_Price", "Demand", "Total_Supply", "Num_Investors", "Num_Production_Sites"],
        "fa": ["Year", "AgentID", "Max_Supply", "Available_Feedstock"],
        "saf_site": ["Year", "Investor_ID", "Production_Output"],
        "investor": ["Year", "ROACE", "Num_Owned_Sites", "AgentID", "Raw_ROACE", "Optimism_Factor"],
        "market_metrics": ["Year"]
    }
 
    # Initialize log collectors
    model_logs, fa_logs, saf_site_logs = [], [], []
    investor_logs, market_metrics_logs = [], []
 
    # Run batch simulations
 
    for i in range(runs):
        seed = (base_seed + i) if base_seed is not None else None
        log_filename = "simulation_batch_run.log"
 
        (
            model_log,
            fa_log,
            saf_site_log,
            investor_log,
            market_metrics_log,
        ) = run_market_model_csv(
            scenario=scenario,
            feedstock_scenario=feedstock_scenario,
            steps=steps,
            log_filename=log_filename,
            config_store=config_store,
            boolean_config_store=boolean_config_store,
            seed=seed,
        )
 
        # Filter and tag logs
        for log, collector, name in zip(
            [model_log, fa_log, saf_site_log, investor_log, market_metrics_log],
            [model_logs, fa_logs, saf_site_logs, investor_logs, market_metrics_logs],
            ["model", "fa", "saf_site", "investor", "market_metrics"]
        ):
            filtered_log = log[keys_to_keep[name]].copy()
            filtered_log["run"] = i
            filtered_log["seed"] = seed
            collector.append(filtered_log)
 
    # Aggregate logs into DataFrames
    def aggregate_logs(logs):
        if logs:
            return pd.concat(logs, ignore_index=True)
        else:
            return pd.DataFrame()
 
    model_df = aggregate_logs(model_logs)
    fa_df = aggregate_logs(fa_logs)
    saf_site_df = aggregate_logs(saf_site_logs)
    investor_df = aggregate_logs(investor_logs)
    market_metrics_df = aggregate_logs(market_metrics_logs)
 
   
    output_dir = os.path.join("output", "batch")
    os.makedirs(output_dir, exist_ok=True)
 
    model_df.to_csv(os.path.join(output_dir, "model_batch_df.csv"), index=False)
    fa_df.to_csv(os.path.join(output_dir, "fa_batch_df.csv"), index=False)
    saf_site_df.to_csv(os.path.join(output_dir, "saf_site_batch_df.csv"), index=False)
    investor_df.to_csv(os.path.join(output_dir, "investor_batch_df.csv"), index=False)
    market_metrics_df.to_csv(os.path.join(output_dir, "market_metrics_batch_df.csv"), index=False)
 
    return (
        model_df,
        fa_df,
        saf_site_df,
        investor_df,
        market_metrics_df
    )