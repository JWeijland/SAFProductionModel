from dash import html, dcc, Input, Output, State

import dash_bootstrap_components as dbc

import dash_daq as daq

from input_stores import (

    BOOLEAN_CONFIG_KEYS,

    SLIDER_CONFIG_KEYS_RANGE,

    SLIDER_CONFIG_KEYS_SINGLE,

)

import ast

import numpy as np

 

import random

 

def generate_boolean_toggles(config_dict):

    return [

        dbc.Card(

            dbc.CardBody(

                [

                    html.Label(

                        key.replace("_", " ").title(),

                        style={"fontWeight": "bold", "marginBottom": "0.5rem"},

                    ),

                    daq.BooleanSwitch(

                        id={"type": "boolean-toggle", "index": key},

                        on=value,

                        label=key.replace("_", " ").title(),

                        labelPosition="top",

                        color="#6ca6cd",

                        style={"width": "100%"},

                    ),

                ]

            ),

            style={

                "borderRadius": "12px",

                "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",

                "backgroundColor": "#f8f9fa",

                "marginBottom": "1rem",

            },

        )

        for key, value in config_dict.items()

    ]

 

def render_booleans_panel():

    return html.Div(

        [

            html.H5("Additional Options", className="text-primary mb-3"),

            dbc.Row(

                [

                    dbc.Col(

                        dbc.Card(

                            dbc.CardBody(

                                [

                                    html.Label(

                                        key.replace("_", " ").title(),

                                        style={

                                            "fontWeight": "bold",

                                            "marginBottom": "0.5rem",

                                        },

                                    ),

                                    daq.BooleanSwitch(

                                        id={"type": "boolean-toggle", "index": key},

                                        on=BOOLEAN_CONFIG_KEYS[key],

                                        color="#0d6efd",  # Bootstrap primary

                                        # label=key.replace("_", " ").title(),

                                        # labelPosition="top",

                                        style={"width": "100%"},

                                    ),

                                ]

                            ),

                            style={

                                "borderRadius": "12px",

                                "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",

                                "backgroundColor": "#f8f9fa",

                            },

                            className="mb-4",

                        ),

                        xs=12,

                        sm=6,

                        md=4,

                        lg=3,

                    )

                    for key in BOOLEAN_CONFIG_KEYS

                ]

            ),

        ],

        style={"padding": "1rem"},

    )

 

def render_config_tab():

    main_cards = [

        render_slider_card(

            "Feedstock & Contracts",

            [

                create_slider("transport_cost", SLIDER_CONFIG_KEYS_SINGLE["transport_cost"], title="Transport Cost (USD)"),

                create_range_slider("feedstock_multiplier", SLIDER_CONFIG_KEYS_RANGE["feedstock_multiplier"], title="Feedstock Multiplier"),

                create_range_slider("contract_percentage", SLIDER_CONFIG_KEYS_RANGE["contract_percentage"], title="Contract Coverage (% of effective capacity)"),

            ],

        ),

        render_slider_card(

            "SAF",

            [

                create_slider("opex", SLIDER_CONFIG_KEYS_SINGLE["opex"], title="Opex (USD)"),

                create_slider("capex_total_cost", SLIDER_CONFIG_KEYS_SINGLE["capex_total_cost"], title="Capex (USD)"),

                create_slider("capex_annual_decrease", SLIDER_CONFIG_KEYS_SINGLE["capex_annual_decrease"], title="Capex Annual Decrease (USD)"),

                create_slider("profit_margin", SLIDER_CONFIG_KEYS_SINGLE["profit_margin"], title="Profit Margin (USD)"),

                create_range_slider("streamday", SLIDER_CONFIG_KEYS_RANGE["streamday"], title="Plant Reliability"),

                create_slider("saf_plant_construction_time", SLIDER_CONFIG_KEYS_SINGLE["saf_plant_construction_time"], title="Construction Time (Years)"),

                create_slider("max_capacity", SLIDER_CONFIG_KEYS_SINGLE["max_capacity"], title="Max Capacity (Tonnes)"),

            ],

        ),

        render_slider_card(

            "Investor",

            [

                create_slider("min_NPV_threshold", SLIDER_CONFIG_KEYS_SINGLE["min_NPV_threshold"], title="Min NPV Threshold (USD)"),

                create_slider("Investment_horizon_length", SLIDER_CONFIG_KEYS_SINGLE["Investment_horizon_length"], title="Investment Horizon (Years)"),

                create_range_slider("Optimism_factor_sample", SLIDER_CONFIG_KEYS_RANGE["Optimism_factor_sample"], title="Optimism Factor"),

            ],

        ),

        render_slider_card(

            "Discount Rate (DR)",

            [

                create_slider("DR_sensitivity_parameter", SLIDER_CONFIG_KEYS_SINGLE["DR_sensitivity_parameter"], title="Sensitivity to ROACE"),

                create_slider("ideal_roace", SLIDER_CONFIG_KEYS_SINGLE["ideal_roace"], title="Ideal ROACE"),

                create_slider("DR_target", SLIDER_CONFIG_KEYS_SINGLE["DR_target"], title="Target DR"),

                create_range_slider("DR", SLIDER_CONFIG_KEYS_RANGE["DR"], title="DR Constraints"),

                create_range_slider("DR_sample", SLIDER_CONFIG_KEYS_RANGE["DR_sample"], title="DR Sample on Initialisation"),

                create_range_slider("ROACE_stability", SLIDER_CONFIG_KEYS_RANGE["ROACE_stability"], title="ROACE Stability"),

            ],

        ),

        render_slider_card(

            "Initialisation",

            [

                create_slider("initial_num_investors", SLIDER_CONFIG_KEYS_SINGLE["initial_num_investors"], title="Number of Investors"),

                create_slider("initial_num_SAF_sites", SLIDER_CONFIG_KEYS_SINGLE["initial_num_SAF_sites"], title="Number of SAF Production Sites"),

                create_slider("start_year", SLIDER_CONFIG_KEYS_SINGLE["start_year"], title="Start Year"),

            ],

        ),

        render_slider_card(

            "Environmental Regulator",

            [

                create_slider("atf_plus_price", SLIDER_CONFIG_KEYS_SINGLE["atf_plus_price"], title="ATF+ Price (USD)"),

                create_slider("blending_mandate", SLIDER_CONFIG_KEYS_SINGLE["blending_mandate"], title="Blending Mandate"),

            ],

        ),

        # CLAUDE START - Phase 2 DIFFERENTIAL ESCALATION: Add price escalation sliders
        render_slider_card(

            "Price Escalation",

            [

                create_slider("inflation_rate", SLIDER_CONFIG_KEYS_SINGLE["inflation_rate"], title="Inflation Rate (CPI)"),

                create_slider("tech_improvement_rate", SLIDER_CONFIG_KEYS_SINGLE["tech_improvement_rate"], title="Tech Improvement Rate"),

                create_slider("market_escalation_rate", SLIDER_CONFIG_KEYS_SINGLE["market_escalation_rate"], title="Market Escalation Rate"),

                create_slider("contract_escalation_rate", SLIDER_CONFIG_KEYS_SINGLE["contract_escalation_rate"], title="Contract Escalation Rate"),

            ],

        ),
        # CLAUDE END - Phase 2 DIFFERENTIAL ESCALATION: Add price escalation sliders

        # CLAUDE START - Phase 3 TIERED PRICING: Add tier configuration sliders
        render_slider_card(

            "Tiered Pricing",

            [

                create_slider("tier_capacity_size", SLIDER_CONFIG_KEYS_SINGLE["tier_capacity_size"], title="Tier Capacity Size (ton/year)"),

                create_slider("tier_1_cost", SLIDER_CONFIG_KEYS_SINGLE["tier_1_cost"], title="Tier 1 Base Cost (USD/ton)"),

                create_slider("tier_cost_increment", SLIDER_CONFIG_KEYS_SINGLE["tier_cost_increment"], title="Tier Cost Increment (USD/ton)"),

                create_slider("aggregator_profit_margin", SLIDER_CONFIG_KEYS_SINGLE["aggregator_profit_margin"], title="Aggregator Profit Margin (USD/ton)"),

            ],

        ),
        # CLAUDE END - Phase 3 TIERED PRICING: Add tier configuration sliders

        # CLAUDE START - TAKE-OR-PAY: Add penalty rate slider
        render_slider_card(

            "Take-or-Pay Contracts",

            [

                create_slider("take_or_pay_penalty_rate", SLIDER_CONFIG_KEYS_SINGLE["take_or_pay_penalty_rate"], title="Penalty Rate (USD/ton)"),

            ],

        ),
        # CLAUDE END - TAKE-OR-PAY: Add penalty rate slider

    ]

 

    # Build the two-column layout explicitly so we can stick Masonry only on the main side

    page = html.Div(

        [

            html.Div(

                # 🔑 Masonry-like flow of cards in the main column

                html.Div(main_cards, className="card-masonry"),

                className="main",

            ),

        ],

        className="layout-grid",

        style={"--side-width": "340px"},

    )

 

    return dbc.Tab(

        label="⚙️ Config",

        tab_id="tab-config",

        children=[

            html.Br(),

            dbc.Row(

                dbc.Col(

                    dbc.Button(

                        "🔄 Reset to Default",

                        id="btn-reset-defaults",

                        color="primary",

                        size="sm",

                        style={"float": "right", "borderRadius": "8px", "backgroundColor": "#6ca6cd"},

                    ),

                    width="auto",

                ),

                justify="end",

                className="mb-3",

            ),

            html.Div(

                [

                    html.H5("Configuration Overrides", className="text-primary"),

                    page,

                    html.Small("Adjust sliders to override default config values.", className="text-muted"),

                ],

                style={"padding": "1rem"},

            ),

        ],

    )

 

seed_input_card = dbc.Card(

    dbc.CardBody(

        [

            html.Label(

                "Seed",

                style={

                    "fontWeight": "500",

                    "fontSize": "0.9rem",

                    "marginBottom": "0.3rem",

                },

            ),

            dcc.Input(

                id="seed-input",

                type="number",

                value=1,

                debounce=True,

                style={

                    "width": "80px",

                    "padding": "0.3rem",

                    "borderRadius": "6px",

                    "border": "1px solid #ccc",

                    "fontSize": "0.9rem",

                    "textAlign": "center",

                },

            ),

        ]

    ),

    style={

        "backgroundColor": "#e6f2fa",

        "borderRadius": "10px",

        "boxShadow": "0 1px 4px rgba(0,0,0,0.1)",

        "padding": "0.5rem",

        "width": "120px",

        "height": "80px",

    },

)

 

# ------------ Model Page Helpers ----------------

# Combine keys

single_keys = list(SLIDER_CONFIG_KEYS_SINGLE.keys())

range_keys = list(SLIDER_CONFIG_KEYS_RANGE.keys())

 

# Create Input objects

inputs = [Input(f"slider-{key}", "value") for key in single_keys + range_keys]

 

slider_outputs = [

    Output(f"slider-{key}", "value") for key in single_keys + range_keys

] + [Output("sliders-store", "data")]





def create_slider(key, config, title=None, description=None, size="md", square=True):

    """

    Single slider in a compact 'square-ish' bubble.

    size: 'sm' | 'md' | 'lg' (smaller tile for side-rail)

    square: True for squarer corners/shape

    """

    classes = ["slider-bubble", f"size-{size}"]

    if square:

        classes.append("squareish")



    return html.Div(

        [

            html.H6(title or key.replace("_", " ").capitalize(), className="slider-title"),

            html.P(description or "", className="text-muted slider-desc"),

            dcc.Slider(

                id=f"slider-{key}",

                min=config["min"],

                max=config["max"],

                step=config.get("step", 1),

                value=config["default"],

                tooltip={"placement": "bottom", "always_visible": True},

                updatemode="drag",

                # Keep it uncluttered—no marks by default:

                marks=config.get("marks", None),

            ),

            html.Div(

                [

                    html.Small(f"{config['min']}", style={"float": "left"}),

                    html.Small(f"{config['max']}", style={"float": "right"}),

                ],

                className="slider-minmax",

            ),

        ],

        className=" ".join(classes) if classes else "slider-bubble",

    )

 

def create_range_slider(key, config, title=None, description=None, size="md", square=True):

    """

    Range slider variant; same compact tile.

    """

    classes = ["slider-bubble", f"size-{size}"]

    if square:

        classes.append("squareish")



    return html.Div(

        [

            html.H6(title or key.replace("_", " ").capitalize(), className="slider-title"),

            html.P(description or "", className="text-muted slider-desc"),

            dcc.RangeSlider(

                id=f"slider-{key}",

                min=config["min"],

                max=config["max"],

                step=config.get("step", 1),

                value=config["default"],

                tooltip={"placement": "bottom", "always_visible": True},

                updatemode="drag",

                marks=config.get("marks", None),   # default to None to avoid clutter

            ),

            html.Div(

                [

                    html.Small(f"{config['min']}", style={"float": "left"}),

                    html.Small(f"{config['max']}", style={"float": "right"}),

                ],

                className="slider-minmax",

            ),

        ],

        className=" ".join(classes) if classes else "slider-bubble",

    )




def render_slider_card(title, sliders, side=False, columns=None):

    """

    side=True -> styles the card for the narrow right rail.

    columns   -> force a fixed column count: 2, 3, or 4 (e.g., Uncertainty needs two rows)

    """

    cls = "section-card mb-3"

    if side:

        cls += " side"

    if columns in (2, 3, 4):

        cls += f" cols-{columns}"

 

    return dbc.Card(

        [

            dbc.CardHeader(

                html.H6(title, className="section-title"),

                className="section-header",

            ),

            dbc.CardBody(

                html.Div(

                    [html.Div(s, className="grid-item") for s in sliders],

                    className="slider-grid",

                )

            ),

        ],

        className=cls,

    )

 

def render_two_column_layout(main_sections, side_sections, side_width="320px"):

    """

    Two-column layout: main content + right side rail (for Poly/Tech).

    """

    return html.Div(

        [

            html.Div(main_sections, className="main"),

            html.Div(side_sections, className="side-rail"),

        ],

        className="layout-grid",

        style={"--side-width": side_width},

    )





# ----------- Output Helpers -------------------

 

def _slugify(s: str) -> str:

    return s.lower().strip().replace(" ", "-")

 

def template_graph(

    title: str,

    graph_id = None,

    figure=None,

    dropdown=None,

    subtitle = None,

    height = 380,

    className: str = "",

):

    if graph_id is None:

        graph_id = f"graph-{_slugify(title)}"

 

    if figure is None:

        figure = {

            "layout": {

                "title": {"text": f"{title}", "x": 0.5, "xanchor": "center"},

                "xaxis": {"visible": False},

                "yaxis": {"visible": False},

                "paper_bgcolor": "#ffffff",

                "plot_bgcolor": "#ffffff",

                "annotations": [

                    {

                        "text": "No data available",

                        "xref": "paper",

                        "yref": "paper",

                        "x": 0.5,

                        "y": 0.5,

                        "showarrow": False,

                        "font": {"color": "#5b6b80"},

                    }

                ],

                "margin": {"l": 40, "r": 30, "t": 50, "b": 40},

            }

        }

 

    header = dbc.CardHeader(

        dbc.Row(

            [

                dbc.Col(

                    html.Div(

                        [

                            html.H5(title, className="card-title mb-0"),

                            (

                                html.Div(subtitle, className="card-subtitle")

                                if subtitle

                                else None

                            ),

                        ],

                        className="graph-card-titles",

                    ),

                ),

                dbc.Col(

                    dbc.Button(

                        "↗ Fullscreen",

                        id={

                            "type": "fs-btn",

                            "target": graph_id,

                        },  # pattern id points to your string graph_id

                        size="sm",

                        className="btn-ghost",

                    ),

                    width="auto",

                    className="graph-card-actions d-flex align-items-center gap-2",

                ),

            ],

            className="g-0 align-items-center justify-content-between",

        ),

        className="graph-card-header",

    )

 

    body_children = []

    if dropdown is not None:

        body_children.append(html.Div(dropdown, className="mb-2"))

 

    body_children.append(

        dcc.Graph(

            id=graph_id,  # <-- string ID (unchanged)

            figure=figure,

            config={

                "displaylogo": False,

                "modeBarButtonsToRemove": [

                    "lasso2d",

                    "select2d",

                    "autoScale2d",

                    "resetScale2d",

                ],

            },

            style={"height": f"{height}px" if isinstance(height, int) else height},

            className="saf-graph",

        )

    )

 

    return dbc.Card(

        [header, dbc.CardBody(body_children, className="graph-card-body")],

        className=f"graph-card mb-4 {className}".strip(),

    )

 

fullscreen_modal = dbc.Modal(

    [

        dbc.ModalHeader(

            dbc.ModalTitle(id="fs-modal-title"),

            close_button=True,  # Optional: adds a built-in close button

        ),

        dbc.ModalBody(dcc.Graph(id="fs-modal-graph")),

        dbc.ModalFooter(

            dbc.Button("Close", id="fs-modal-close", className="ms-auto", n_clicks=0)

        ),

        dcc.Store(id="fs-modal-fig-store"),

    ],

    id="fs-modal",

    is_open=False,

    size="fullscreen",

    centered=True,

    scrollable=True,

)

 

# -------------- Callback Helpers -------------------------

 

def _extract_prices_from_sequence(obj):

 

    if not isinstance(obj, (list, tuple)):

        return None

    prices = []

    for item in obj:

        if isinstance(item, (int, float)):

            prices.append(float(item))

        elif isinstance(item, (list, tuple)) and item:

            first = item[0]

            if isinstance(first, (int, float)):

                prices.append(float(first))

    return prices if prices else None

 

def _parse_list(s):

    """

    Parse a stringified or native list forecast to list[float].

    """

    if s is None or (isinstance(s, float) and np.isnan(s)):

        return None

    if isinstance(s, list):

        return _extract_prices_from_sequence(s)

    if isinstance(s, str):

        s = s.strip()

        if not s:

            return None

        try:

            obj = ast.literal_eval(s)

            return _extract_prices_from_sequence(obj)

        except Exception:

            return None

    return None

 

# ----------- Feedstock Reliability KPIs ----------------

 

import pandas as pd

 

def _prep_runs(df):

    """Coerce numerics, build run (run|seed if seed exists), filter required cols."""

    for c in ["Year", "Demand", "Total_Supply", "run"]:

        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "seed" in df.columns:

        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

        df["run"] = (

            df["run"].astype("Int64").astype(str)

            + "|"

            + df["seed"].astype("Int64").astype(str)

        )

        df = df.dropna(subset=["Year", "Demand", "Total_Supply", "run", "seed"])

    else:

        df["run"] = df["run"].astype("Int64").astype(str)

        df = df.dropna(subset=["Year", "Demand", "Total_Supply", "run"])

    return df

 

def _compute_feed_reliability_per_run(df):

    """Return per-run reliability (fraction in [0,1]) and overall agg (mean, p10, p90)."""

    # Boolean success per row

    df = df.copy()

    df["reliable"] = (df["Total_Supply"] > df["Demand"]).astype(float)

 

    # Per run & year (in case of duplicates), then average across years in that run

    per_run_year = df.groupby(["run", "Year"], as_index=False)["reliable"].mean()

    per_run = (

        per_run_year.groupby("run")["reliable"]

        .mean()

        .reset_index(name="reliability")

    )

 

    # Aggregate across runs

    agg = (

        per_run["reliability"]

        .agg(

            mean="mean",

            p10=lambda s: s.quantile(0.10, interpolation="linear"),

            p90=lambda s: s.quantile(0.90, interpolation="linear"),

        )

        .to_dict()

    )

 

    return per_run, agg






fullscreen_modal = dbc.Modal(

    [

        dbc.ModalHeader(

            dbc.ModalTitle(id="fs-modal-title"),

            close_button=True,  # Optional: adds a built-in close button

        ),

        dbc.ModalBody(dcc.Graph(id="fs-modal-graph")),

        dbc.ModalFooter(

            dbc.Button("Close", id="fs-modal-close", className="ms-auto", n_clicks=0)

        ),

        dcc.Store(id="fs-modal-fig-store"),

    ],

    id="fs-modal",

    is_open=False,

    size="fullscreen",

    centered=True,

    scrollable=True,

)





MAX_SEED = 2**32 - 1

 

def _rand_seed():

    # OS-backed randomness; avoids predictable PRNG sequences

    return random.SystemRandom().randrange(0, MAX_SEED)