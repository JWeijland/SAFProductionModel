 

from __future__ import annotations

 

import dash

import dash_bootstrap_components as dbc

from dash import html, dcc, register_page

 

# Register this as the Home page (root path)

register_page(

    __name__,

    path="/",

    name="Home",

    title="SAF Market Model • Home",

    order=1,

)

 

# Simple landing layout with brief intro and quick links

layout = dbc.Container(

    [

        # Page header uses your theme classes

        html.Div(

            [

                html.H1("SAF Market Model", className="page-title"),

                html.P(

                    "Simulate, analyse, and optimise the Sustainable Aviation Fuel Indian Market.",

                    className="page-subtitle",

                ),

            ],

            className="page-header",

        ),

        # Intro/hero callout – styled like your graph cards for visual consistency

        dbc.Card(

            dbc.CardBody(

                html.P(

                    "Configure inputs, run simulations, and explore results to see how "

                    "feedstock costs, investor activity, and policy incentives shape SAF "

                    "supply, investment, and pricing.",

                    className="mb-0",

                )

            ),

            className="graph-card home-hero",

        ),

        html.H3(

            "Quick guide", className="mt-3 mb-2", style={"color": "var(--fs-primary)"}

        ),

        html.Ul(

            [

                html.Li(

                    [

                        html.Strong("Model setup: "),

                        "Load default parameters and tweak for experiments.",

                    ]

                ),

                html.Li(

                    [

                        html.Strong("Run simulation: "),

                        "Start a run from the Model page: Either a single run or multiple replications.",

                    ]

                ),

                html.Li(

                    [

                        html.Strong("View outputs: "),

                        "Review price trajectories, production volumes, and compare scenarios.",

                    ]

                ),

                html.Li(

                    [

                        html.Strong("View Multiple Replication outputs: "),

                        "Explore higher level KPIs for multiple instances of the simulation.",

                    ]

                ),

            ],

            className="mb-3",

        ),

        dbc.Card(

            dbc.CardBody(

                [

                    html.H4("Model Summary", className="card-title", style={"color": "#0c72b6", "fontWeight": "600"}),

                    html.P(

                        "The SAF Market Model is an agent-based simulation of India’s transition to Sustainable Aviation Fuel (SAF). "

                        "It focuses on how investor decisions — influenced by policy, feedstock availability, and market conditions — "

                        "shape SAF production, pricing, and adoption over time.",

                        className="card-text",

                    ),

                    html.Ul(

                        [

                            html.Li(

                                "Models investor behavior using NPV and ROACE-based logic."

                            ),

                            html.Li(

                                "Simulates feedstock variability and competition across states."

                            ),

                            html.Li(

                                "Forecasts SAF prices using market merit order and demand matching."

                            ),

                            html.Li(

                                "Supports scenario analysis for policy, market, and feedstock outputs."

                            ),

                        ],

                        className="card-text",

                    ),

                ]

            ),

            className="graph-card model-summary",

        ),

        html.Img(

            src="/assets/productflowdiagram.jpg",

            style={"width": "60%", "borderRadius": "10px"},

        ),

        # Tip callout – soft panel look aligned with your tokens

        dbc.Alert(

            [

                html.Span("Tip: ", className="fw-bold"),

                "Use the ",

                html.Strong("Model"),

                " page to customize inputs and start your run. Results will appear in ",

                html.Strong("Outputs"),

                " or ",

                html.Strong("Multiple Replications"),

                " when the simulation completes.",

            ],

            color="light",

            className="home-tip",

        ),

    ],

    fluid=True,

    className="page content-wrap",

)

 

 

