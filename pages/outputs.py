 

from __future__ import annotations

import dash

import dash_bootstrap_components as dbc

from dash import html, dcc, register_page

from helper import fullscreen_modal

 

register_page(

    __name__,

    path="/outputs",

    name="Outputs",

    title="SAF Market Model • Outputs",

    order=4,

)

 

def _slugify(s: str) -> str:

    return s.lower().strip().replace(" ", "-")

 

def template_graph(

    title: str,

    graph_id: str | None = None,

    figure=None,

    dropdown=None,

    subtitle: str | None = None,

    height: int | str = 380,

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






layout = html.Div(

    [

        dcc.Store(id="store-graph-saf-price-over-time", storage_type="session"),

        dcc.Store(id="store-graph-supply-demand", storage_type="session"),

        dcc.Store(id="store-consumer-price-forecast-graph", storage_type="session"),

        dcc.Store(id="store-graph-owned-sites-per-investor", storage_type="session"),

        dcc.Store(id="store-plant-ebit-graph", storage_type="session"),

        dcc.Store(id="store-investor-kpis-graph", storage_type="session"),

        dcc.Store(id="store-graph-supply-by-plant", storage_type="session"),

        dcc.Store(id="store-feedstock-availability-graph", storage_type="session"),

        # Additional Metrics stores

        dcc.Store(id="store-graph-contract-vs-spot-prices", storage_type="session"),

        dcc.Store(id="store-graph-cumulative-penalties", storage_type="session"),

        dcc.Store(id="store-graph-tier-allocation-by-state", storage_type="session"),

        # dcc.Store(id="store-fs-modal-graph", storage_type="session"),

        dbc.Container(

            [

                html.Div(

                    [

                        dcc.Store(id="store-logs"),  # Store for logs

                        html.H5("Simulation Logs"),

                        html.Pre(

                            id="log-output",

                            style={

                                "whiteSpace": "pre-wrap",

                                "maxHeight": "400px",

                                "overflowY": "scroll",

                            },

                        ),

                    ]

                ),

                html.H2("Outputs"),

                html.P(

                    "Explore the results of the SAF market model across different dimensions."

                ),

                dbc.Tabs(

                    [

                        dbc.Tab(

                            label="Market Metrics",

                            tab_id="market",

                            children=[

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "SAF Price Over Time",

                                                graph_id="graph-saf-price-over-time",

                                            ),

                                            md=6,

                                        ),

                                        dbc.Col(

                                            template_graph(

                                                "Capacity vs Demand",

                                                graph_id="graph-supply-demand",

                                            ),

                                            md=6,

                                        ),

                                    ]

                                ),

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Consumer Price Forecast over Time",

                                                graph_id="consumer-price-forecast-graph",

                                            ),

                                            md=12,

                                        ),

                                    ]

                                ),

                            ],

                        ),

                        dbc.Tab(

                            label="Econometric Metrics",

                            tab_id="investor",

                            children=[

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Owned sites per Investor",

                                                graph_id="graph-owned-sites-per-investor",

                                            ),

                                            md=6,

                                        ),

                                        dbc.Col(

                                            template_graph(

                                                "Production per Plant vs Demand",

                                                graph_id="graph-production-by-investor-vs-demand",

                                            ),

                                            md=6,

                                        ),

                                        dbc.Col(

                                            template_graph(

                                                "Take-or-Pay: Curtailed Volume & Penalty Cost",

                                                graph_id="graph-curtailed-volume-by-investor",

                                            ),

                                            md=6,

                                        ),

                                    ]

                                ),

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Investor KPIs",

                                                graph_id="investor-kpis-graph",

                                                dropdown=dcc.Dropdown(

                                                    id="investor-metrics-dropdown",

                                                    placeholder="Select an investor",

                                                ),

                                            ),

                                            md=12,

                                        ),

                                    ]

                                ),

                                dbc.Row(

                                    dbc.Col(

                                        template_graph(

                                            "Plant EBITs Over Time",

                                            graph_id="plant-ebit-graph",

                                            dropdown=dcc.Dropdown(

                                                id="dropdown-selected-plants",

                                                options=[],  # Will be populated dynamically

                                                multi=True,

                                                placeholder="Select plants to display",

                                            ),

                                        ),

                                        md=12,

                                    )

                                ),

                            ],

                        ),

                        dbc.Tab(

                            label="Production Metrics",

                            tab_id="production",

                            children=[

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "SAF Supply by Plant",

                                                graph_id="graph-supply-by-plant",

                                            ),

                                            md=6,

                                        ),

                                        dbc.Col(

                                            template_graph(

                                                "Feedstock Availability per State",

                                                graph_id="feedstock-availability-graph",

                                                dropdown=dcc.Dropdown(

                                                    id="state-dropdown",

                                                    options=[],

                                                    placeholder="Select a state",

                                                ),

                                            ),

                                            md=6,

                                        ),

                                    ]

                                ),

                            ],

                        ),

                        dbc.Tab(

                            label="Additional Metrics",

                            tab_id="additional",

                            children=[

                                # Section 1: Price Dynamics

                                html.H4("Price Dynamics & Contract Analysis", className="mt-3 mb-3", style={"color": "#0c72b6"}),

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Contract vs Spot Feedstock Prices",

                                                subtitle="Contract: locked tier price with CPI escalation. Spot: current market tier price (rises as capacity fills)",

                                                graph_id="graph-contract-vs-spot-prices",

                                            ),

                                            md=12,

                                        ),

                                    ]

                                ),



                                # Section 2: Take-or-Pay Penalties

                                html.Hr(className="my-4"),

                                html.H4("Take-or-Pay Penalties", className="mb-3", style={"color": "#0c72b6"}),

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Cumulative Penalties by Plant (Ranked)",

                                                subtitle="Total take-or-pay penalties: which plants bear the highest curtailment cost?",

                                                graph_id="graph-cumulative-penalties",

                                            ),

                                            md=12,

                                        ),

                                    ]

                                ),


                                # Section 3: Tier Allocation Across States

                                html.Hr(className="my-4"),

                                html.H4("Tier Allocation & Competitive Dynamics", className="mb-3", style={"color": "#0c72b6"}),

                                html.P(
                                    "Track how feedstock tier capacity is allocated across states over time. Shows which states fill first (lowest tier prices attract early investment) and tier progression.",
                                    style={"marginBottom": "20px", "fontSize": "14px", "color": "#666"}
                                ),

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Cumulative Tier Allocation by State",

                                                subtitle="Shows capacity allocation evolution: states with lowest tiers fill first (early-mover advantage)",

                                                graph_id="graph-tier-allocation-by-state",

                                                height=450,

                                            ),

                                            md=12,

                                        ),

                                    ]

                                ),


                                # Section 4: KPI Comparison

                                html.Hr(className="my-4"),

                                html.H4("Model Comparison: Baseline vs Advanced", className="mb-3", style={"color": "#0c72b6"}),

                                html.P(
                                    "Comparison of key metrics between baseline and advanced model.",
                                    style={"marginBottom": "20px", "fontSize": "14px", "color": "#666"}
                                ),

                                dbc.Row([
                                    dbc.Col(
                                        html.Div(id="kpi-comparison-table"),
                                        md=12,
                                    ),
                                ]),

                            ],

                        ),

                    ],

                    id="output-tabs",

                    active_tab="market",

                ),

            ],

            fluid=True,

        ),

        html.Div(

            id="feedstock-widget",

            children=[

                dbc.Button(

                    "Feedstock prices",

                    id="feedstock-toggle",

                    n_clicks=0,

                    color="link",  # we’ll skin it via CSS

                    className="feedstock-toggle-btn",

                ),

                dbc.Collapse(

                    id="feedstock-collapse",

                    is_open=False,

                    children=dbc.Card(

                        dbc.CardBody(

                            [

                                dbc.ListGroup(

                                    id="feedstock-list",

                                    className="feedstock-list",

                                    flush=True,

                                ),

                            ]

                        ),

                        className="feedstock-card",

                    ),

                ),

            ],

            className="feedstock-widget",

        ),

        fullscreen_modal,

    ]

)

 

 


