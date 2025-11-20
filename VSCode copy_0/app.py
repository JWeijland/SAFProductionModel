from __future__ import annotations

import dash

import dash_bootstrap_components as dbc

from dash import html, dcc

import os

 

app = dash.Dash(

    __name__,

    use_pages=True,

    pages_folder="pages",

    suppress_callback_exceptions=True,

    external_stylesheets=[dbc.themes.BOOTSTRAP],

    title="SAF Market Model",

)

 

server = app.server

from callbacks import register_callbacks

SCENARIOS = ["Surge", "Horizon", "Archipelagos"]

 

from dash.dependencies import Input, Output, ALL, ClientsideFunction

 

app.clientside_callback(

    ClientsideFunction(namespace="fullscreen", function_name="open"),

    Output("fs-modal", "is_open"),

    Output("fs-modal-title", "children"),

    Output("fs-modal-fig-store", "data"),

    Input({"type": "fs-btn", "target": ALL}, "n_clicks"),

    Input("fs-modal-close", "n_clicks"),

    prevent_initial_call=True,

)

 

def build_navbar() -> dbc.Navbar:

    pages = sorted(dash.page_registry.values(), key=lambda p: p.get("order", 999))

    nav_items = [

        dbc.NavItem(

            dbc.NavLink(

                p["name"],

                href=app.get_relative_path(p["path"]),              

                active="exact",

                className="px-3 py-2 rounded nav-link"

            ),

            className="me-2"

        )

        for p in pages

        if not p.get("is_hidden", False)

    ]

 

    return dbc.Navbar(

        dbc.Container([

            html.A(

                dbc.Row([

                    dbc.Col(html.Span("🛫", className="me-2", style={"fontSize": "1.5rem"})),

                    dbc.Col(dbc.NavbarBrand("SAF Market Model", className="ms-0", style={"fontSize": "1.5rem", "fontWeight": "500"})),

                ], align="center", className="g-0"),

                href="/",

                className="text-decoration-none"

            ),

            dbc.NavbarToggler(id="navbar-toggler"),

            dbc.Collapse(

                dbc.Nav(nav_items, className="ms-auto", navbar=True),

                id="navbar-collapse",

                navbar=True

            ),

        ], fluid=True),

        color="light",

        dark=False,

        className="mb-3 shadow-sm rounded",

        style={

            "borderRadius": "12px",

            "background": "linear-gradient(to right, #dbeeff, #f0f8ff)"

        }

    )

 

# ---------- Layout ----------

app.layout = dbc.Container(

    fluid=True,

    children=[

        build_navbar(),

 

        # Global stores of results

        dcc.Store(id="store-booleans", storage_type="session"),

        dcc.Store(id="store-config", storage_type="session"),

        dcc.Store(id="store-states", storage_type="session"),

        dcc.Store(id="store-demand", storage_type="session"),

 

        dcc.Store(id="store-model", storage_type="memory"),

        dcc.Store(id="store-investor", storage_type="memory"),

        dcc.Store(id="store-feedstock-aggregator", storage_type="memory"),

        dcc.Store(id="store-saf-production-site", storage_type="memory"),

        dcc.Store(id="store-market-metric-log", storage_type="memory"),

        dcc.Store(id="store-batch-model", storage_type="memory"),

        dcc.Store(id="store-batch-investor", storage_type="memory"),

        dcc.Store(id="store-batch-feedstock-aggregator", storage_type="memory"),

        dcc.Store(id="store-batch-saf-production-site", storage_type="memory"),

        dcc.Store(id="store-batch-market-metric-log", storage_type="memory"),

 

        dcc.Store(id="store-graph-saf-price-over-time", storage_type="session"),

        dcc.Store(id="store-graph-supply-demand", storage_type="session"),

        dcc.Store(id="store-consumer-price-forecast-graph", storage_type="session"),

        dcc.Store(id="store-graph-owned-sites-per-investor", storage_type="session"),

        dcc.Store(id="store-plant-ebit-graph", storage_type="session"),

        dcc.Store(id="store-investor-kpis-graph", storage_type="session"),

        dcc.Store(id="store-graph-supply-by-plant", storage_type="session"),

        dcc.Store(id="store-feedstock-availability-graph", storage_type="session"),

 

           

        # Render whichever page is active (Intro, Model, etc.)

        dash.page_container,

    ],

)

 

# ---------- Entrypoint ----------

if __name__ == "__main__":

    register_callbacks(app)

    app.run(debug=True)

else:

    register_callbacks(app)

