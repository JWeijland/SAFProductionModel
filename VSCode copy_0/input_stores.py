SLIDER_CONFIG_KEYS_SINGLE = {

    "min_NPV_threshold": {"min": 0, "max": 100000, "step": 1000, "default": 0.0},

    "opex": {"min": 0, "max": 571.42, "step": 28.57, "default": 285.71},

    "transport_cost": {"min": 0, "max": 1200.0, "step": 60.0, "default": 600.0},

    "profit_margin": {"min": 0, "max": 500.0, "step": 20.0, "default": 250.0},

    "initial_num_investors": {"min": 0, "max": 10, "step": 1, "default": 1.0},

    "initial_num_SAF_sites": {"min": 0, "max": 10, "step": 1, "default": 1.0},

    "max_capacity": {"min": 0, "max": 200000.0, "step": 10000.0, "default": 100000.0},

    "saf_plant_construction_time": {"min": 0, "max": 10, "step": 1, "default": 4.0},

    "Investment_horizon_length": {"min": 0, "max": 50.0, "step": 1, "default": 25.0},

    "DR_sensitivity_parameter": {"min": 0, "max": 1, "step": 0.01, "default": 0.2},

    "ideal_roace": {"min": 0, "max": 1, "step": 0.01, "default": 0.075},

    "DR_target": {"min": 0, "max": 1, "step": 0.01, "default": 0.075},

    "capex_total_cost": {"min": 0, "max": 1000000000.0, "step": 10000000, "default": 500000000.0},

    "atf_plus_price": {"min": 0, "max": 4000.0, "step": 200.0, "default": 2000.0},

    "capex_annual_decrease": {"min": 0, "max": 1, "step": 0.01, "default": 0.01},

    "blending_mandate": {"min": 0, "max": 1, "step": 0.01, "default": 0.01},

    "start_year": {"min": 0, "max": 4048.0, "step": 202.4, "default": 2024.0}

}

 

SLIDER_CONFIG_KEYS_RANGE = {

    "feedstock_multiplier": {

        "min": 0, "max": 2, "step": 0.05, "default": [0.8, 1.2]

    },

    "DR_sample": {

        "min": 0, "max": 0.5, "step": 0.01, "default": [0.06, 0.12]

    },

    "Optimism_factor_sample": {

        "min": 0.5, "max": 1.5, "step": 0.1, "default": [1, 1]

    },

    "DR": {

        "min": 0.01, "max": 0.5, "step": 0.01, "default": [0.04, 0.2]

    },

    "streamday": {

        "min": 0, "max": 1, "step": 0.01, "default": [0.95, 0.98]

    },

    "ROACE_stability": {

        "min": 0, "max": 1, "step": 0.01, "default": [0.06, 0.09]

    }

}








SCENARIOS = ["Surge", "Horizon", "Archipelagos"]

FEEDSTOCK_SCENARIOS = ["Surplus", "Limited"]

 

BOOLEAN_CONFIG_KEYS = {

    "operational_initially": True,

    "capex_decrease": False,

    # Add more here as needed

}

