from src.Model import SAFMarketModel
import logging
import pandas as pd
 
log_filename = "simulation_run.log"
 
Scenario = "Surge"
# Scenario = "Horizon"
# Scenario = "Archipelagos"
 
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.DEBUG,
    format="%(levelname)s - %(message)s",
)
 
# Unpack input data from CSV files
boolean_df = pd.read_csv("input/booleans.csv")
config_df = pd.read_csv("input/config.csv")
states_df = pd.read_csv("input/states_data_undersupply.csv")
demand_df = pd.read_csv(f"input/{Scenario.lower()}_demand_values.csv")
 
# Strip whitespace from headers
boolean_df.columns = boolean_df.columns.str.strip()
config_df.columns = config_df.columns.str.strip()
states_df.columns = states_df.columns.str.strip()
demand_df.columns = demand_df.columns.str.strip()
 
booleans = dict(zip(boolean_df["key"], boolean_df["value"]))
demand = dict(zip(demand_df["year"], demand_df["value"]))
config = dict(zip(config_df["key"], config_df["value"]))
states_data = {
    row["state"]: {
        "max_supply": row["max_supply (Tonnes/year)"],
        "feedstock_price": row["feedstock_price(USD/Tonne)"],
        "feedstock_type": row["feedstock_type"],
    }
    for _, row in states_df.iterrows()
}
 
if __name__ == "__main__":
    """
    Main execution block for the SAF Market Model simulation.
    Loads demand forecast, initializes the model, and runs the simulation for a fixed number of steps.
    Logs market price and simulation progress at each step.
 
    """
    # Initialise model
    model = SAFMarketModel(
        config=config,
        states_data=states_data,
        atf_demand_forecast=demand,
        booleans=booleans,
    )
 
    # Run for X steps
    for step in range(1, 101):
        logging.info(f"\n--- Step {step} ---\n")
        print(f"--- Step {step} ---")
        model.step()
        logging.info(f"Market price: {model.market_price}")
       
    model.export_logs()
 
    # After simulation run, export DataCollector results to output folder
    model_vars = model.datacollector.get_model_vars_dataframe()
    agent_vars = model.datacollector.get_agent_vars_dataframe()
 
    model_vars.to_csv("output/model_vars.csv")
    agent_vars.to_csv("output/agent_vars.csv")
    logging.info("Simulation data exported to output/model_vars.csv and output/agent_vars.csv")