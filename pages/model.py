
from __future__ import annotations

import dash_bootstrap_components as dbc

from dash import html, dcc, register_page

from input_stores import (

    FEEDSTOCK_SCENARIOS,

    SCENARIOS,

)

 

from helper import (

    generate_boolean_toggles,

    render_booleans_panel,

    render_config_tab,

    _rand_seed,

    MAX_SEED,

)

 

register_page(

    __name__,

    path="/saf-market-model/model",

    name="Model",

    title="SAF Market Model • Model",

    order=2,

)

 

layout = dbc.Container(

    fluid=True,

    children=[

        html.Div(id="dummy-trigger", children="init", style={"display": "none"}),

        dcc.Store(id="sliders-store", storage_type="session", data={}),

        dcc.Store(id="config-store", storage_type="session"),

        dcc.Store(id="boolean-config-store", data=None),

        html.Br(),

        dbc.Row(

            [

                dbc.Col(

                    dbc.Card(

                        dbc.CardBody(

                            [

                                # --- Top controls (unchanged) ---

                                html.Label("Scenario (ATF Demand)"),

                                dcc.Dropdown(

                                    id="scenario-dd",

                                    options=[

                                        {"label": s, "value": s} for s in SCENARIOS

                                    ],

                                    value="Surge",

                                    clearable=False,

                                ),

                                html.Br(),

                                html.Label("Years"),

                                dcc.Slider(

                                    id="steps-slider",

                                    min=10,

                                    max=200,

                                    step=10,

                                    value=100,

                                    marks={i: str(i) for i in [10, 20, 50, 100, 200]},

                                ),

                                html.Br(),

                                # --- Bottom section: Feedstock (left) + Seed mini-card (right) ---

                                html.Hr(style={"margin": "0.75rem 0"}),

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            [

                                                html.Label(

                                                    "Feedstock Scenario",

                                                    className="mb-1",

                                                ),

                                                dcc.Dropdown(

                                                    id="feedstock-scenario-dd",

                                                    options=[

                                                        {"label": s, "value": s}

                                                        for s in FEEDSTOCK_SCENARIOS

                                                    ],

                                                    value="Oversupply",

                                                    clearable=False,

                                                    style={"fontSize": "0.9rem"},

                                                ),

                                            ],

                                            xs=12,

                                            md=8,

                                            className="mb-2 mb-md-0",

                                        ),

                                        dbc.Col(

                                            dbc.Card(

                                                dbc.CardBody(

                                                    [

                                                        html.Label(

                                                            "Seed",

                                                            style={

                                                                "fontWeight": "500",

                                                                "fontSize": "0.8rem",

                                                                "marginBottom": "0.2rem",

                                                                "textAlign": "center",

                                                            },

                                                        ),

                                                        dcc.Input(

                                                            id="seed-input",

                                                            type="number",

                                                            value=_rand_seed(),  # <-- default is random

                                                            min=0,

                                                            max=MAX_SEED,

                                                            step=1,

                                                            debounce=True,

                                                            persistence=True,

                                                            persistence_type="session",

                                                            style={

                                                                "width": "60px",

                                                                "padding": "0.2rem",

                                                                "borderRadius": "6px",

                                                                "border": "1px solid #ccc",

                                                                "fontSize": "0.8rem",

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

                                                    "height": "80px",

                                                    "width": "100px",

                                                    "marginTop": "auto",

                                                    "marginBottom": "auto",

                                                },

                                            ),

                                            xs=12,

                                            md=4,

                                            className="d-flex justify-content-md-end",

                                        ),

                                    ],

                                    className="g-2 align-items-start",

                                ),

                            ],

                            className="d-flex flex-column",  # enables optional sticky-bottom

                        ),

                        style={

                            "backgroundColor": "#f8f9fa",

                            "borderRadius": "10px",

                            "boxShadow": "0 1px 4px rgba(0,0,0,0.1)",

                            "padding": "1rem",

                        },

                    ),

                    md=5,

                ),

                # Right: Run button

                dbc.Col(

                    [

                        dbc.Button(

                            "🚀 Run Simulation",

                            id="btn-run",

                            style={

                                "backgroundColor": "#6ca6cd",

                                "border": "none",

                                "color": "white",

                                "padding": "1rem 2rem",

                                "fontSize": "1.2rem",

                                "fontWeight": "500",

                                "borderRadius": "8px",

                                "boxShadow": "0 2px 6px rgba(0,0,0,0.2)",

                            },

                            size="lg",

                        ),

                        html.Div(id="run-status", className="mt-2"),

                        dbc.Button(

                            "▶️ Run Multiple Replications",

                            id="btn-batch-run",

                            style={

                                "backgroundColor": "#0c72b6",

                                "border": "none",

                                "color": "white",

                                "padding": "0.6rem 1.2rem",

                                "fontSize": "1rem",

                                "fontWeight": "500",

                                "borderRadius": "6px",

                                "boxShadow": "0 2px 4px rgba(0,0,0,0.15)",

                            },

                            size="md",

                        ),

                        dcc.Input(

                            id="runs-input", type="number", value=10, min=1, step=1

                        ),

                    ],

                    md=5,

                    className="d-flex flex-column align-items-center justify-content-center",

                ),

            ],

            className="mb-3",

        ),

        html.Hr(),

        # Tabs Section

        dbc.Tabs(

            id="tabs",

            active_tab="tab-config",

            children=[

                dbc.Tab(

                    label="Config",

                    tab_id="tab-config",

                    children=[

                        dbc.Row(

                            [

                                dbc.Col(

                                    render_config_tab(),  # Main config content

                                    width=9,  # Adjust width as needed

                                ),

                                dbc.Col(

                                    render_booleans_panel(),  # Boolean switches

                                    width=3,  # Thinner column

                                    style={

                                        "borderLeft": "1px solid #ddd",

                                        "paddingLeft": "15px",

                                    },

                                ),

                            ]

                        )

                    ],

                ),

 

            ],

        ),

    ],

)

 

 

 