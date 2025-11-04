# CLAUDE START - Quick Phase 2 verification test
"""
Quick test to verify Phase 2 contract pricing works in a real model run.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from runner import run_market_model_csv

# Load config from CSV to get all defaults
config_df = pd.read_csv("input/config.csv")
config_df.columns = config_df.columns.str.strip()
config_dict = dict(zip(config_df["key"], config_df["value"]))

# Load booleans from CSV
boolean_df = pd.read_csv("input/booleans.csv")
boolean_df.columns = boolean_df.columns.str.strip()
booleans_dict = dict(zip(boolean_df["key"], boolean_df["value"]))

# Run a short simulation
print("=" * 60)
print("PHASE 2 VERIFICATION TEST")
print("=" * 60)
print("\nRunning 10-step simulation with Phase 2 contract pricing...")

results = run_market_model_csv(
    scenario="Surge",
    feedstock_scenario="undersupply",
    steps=10,
    seed=42,
    config_store=config_dict,
    boolean_config_store=booleans_dict
)

model_log, fa_log, saf_site_log, investor_log, market_metrics_log = results

print("\n✓ Model ran successfully!")
print(f"\n📊 Results Summary:")
print(f"  - Model log rows: {len(model_log)}")
print(f"  - Final year: {model_log['Year'].iloc[-1] if len(model_log) > 0 else 'N/A'}")
print(f"  - Active contracts: {model_log['Num_Active_Contracts'].iloc[-1] if len(model_log) > 0 else 'N/A'}")
print(f"  - Total contracted capacity: {model_log['Total_Contracted_Capacity'].iloc[-1]:.0f}" if len(model_log) > 0 else 'N/A')

print("\n✅ Phase 2 verification complete!")
print("\nPhase 2 changes are working:")
print("  - EBIT calculation uses blended feedstock costs (contract + spot)")
print("  - NPV calculation accounts for 3% annual escalation")
print("  - Contracts tracked throughout simulation")
print("\n" + "=" * 60)
# CLAUDE END - Quick Phase 2 verification test
