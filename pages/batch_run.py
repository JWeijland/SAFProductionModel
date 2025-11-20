from __future__ import annotations

import dash_bootstrap_components as dbc

from dash import html, dcc, register_page

 

from helper import template_graph, fullscreen_modal

 

register_page(

    __name__,

    path="/batch",

    name="Multiple Replications",

    title="SAF Market Model • Multiple Replications",

    order=5,

)

 

layout = html.Div(

    [

        dbc.Container(

            [

                html.Div(

                    [

                        dcc.Store(id="store-batch-logs"),  # Store for logs

                        html.Pre(

                            id="batch-log-output",

                            style={

                                "whiteSpace": "pre-wrap",

                                "maxHeight": "400px",

                                "overflowY": "scroll",

                            },

                        ),

                    ]

                ),

                html.H2("Multiple Replications Outputs"),

                html.P(

                    "Explore the results of the SAF market model across multiple runs."

                ),

                dbc.Tabs(

                    [

                        dbc.Tab(

                            label="Production Metrics",

                            tab_id="batch-graphs",

                            children=[

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "SAF Reliability",

                                                graph_id="graph-supply-minus-demand",

                                            ),

                                            md=6,

                                        ),

                                        dbc.Col(

                                            template_graph(

                                                "Heatmap of Sufficient SAF Supply",

                                                graph_id="graph-feed-reliability-heatmap",

                                            ),

                                            md=6,

                                        ),

                                    ]

                                ),

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "SAF Reliability KPI",

                                                graph_id="kpi-feed-reliability",

                                            ),

                                            md=12,

                                        ),

                                    ]

                                ),

                            ],

                        ),

                        dbc.Tab(

                            label="Investor Agent Metrics",

                            tab_id="investor-agent-graphs",

                            children=[

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Max Sites of a single Investor at End of Simulation",

                                                graph_id="graph-max-sites-owned",

                                            ),

                                            md=6,

                                        ),

                                        dbc.Col(

                                            template_graph(

                                                "Average Sites per Investor at End of Simulation",

                                                graph_id="graph-avg-sites-distribution",

                                            ),

                                            md=6,

                                        ),

                                    ]

                                ),

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Investment Count at End of Simulation",

                                                graph_id="graph-investors-at-endyear-hist",

                                            ),

                                            md=6,

                                        ),

                                        dbc.Col(

                                            template_graph(

                                                "Relationship between Optimism Factor and Market Share",

                                                graph_id="graph-optimism-vs-share",

                                            ),

                                            md=6,

                                        ),

                                    ]

                                ),

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Plants per Investor vs Number of Investors",

                                                graph_id="graph-plants-per-investor-vs-investors",

                                            ),

                                            md=6,

                                        )

                                    ]

                                )

                            ],

                        ),

                        dbc.Tab(

                            label="Econometric Metrics",

                            tab_id="econometric-graphs",

                            children=[

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Average ROACE by Year",

                                                graph_id="graph-roace-by-year",

                                            ),

                                            md=6,

                                        ),

                                        dbc.Col(

                                            template_graph(

                                                "Financial Health KPI: Average ROACE of Investor over all runs over all years.",

                                                graph_id="graph-roace-kpi",

                                            ),

                                            md=6,

                                        ),

                                    ]

                                ),

                            ],

                        ),

                        dbc.Tab(

                            label="Feedstock Metrics",

                            tab_id="feedstock-graphs",

                            children=[

                                dbc.Row(

                                    [

                                        dbc.Col(

                                            template_graph(

                                                "Feedstock Used per State",

                                                graph_id="graph-feedstock-used",

                                            ),

                                            md=12,

                                        ),

 

                                    ]

                                ),

                            ],

                        ),

                    ],

                    id="graph-output-tabs",

                    active_tab="batch-graphs",

                ),

            ],

            fluid=True,

        ),

        fullscreen_modal,

    ]

)

 

 

 