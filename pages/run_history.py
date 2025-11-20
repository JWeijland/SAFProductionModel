"""
Run History Page - View and manage saved simulation runs
"""
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, register_page, callback, Input, Output, State, no_update
from dash.dependencies import ALL
import pandas as pd
from run_manager import get_run_manager

register_page(
    __name__,
    path="/run-history",
    name="Run History",
    title="SAF Market Model • Run History",
    order=3,
)


def create_run_card(run_data):
    """Create a card for displaying a single run"""
    run_id = run_data.get("run_id", "Unknown")
    run_name = run_data.get("run_name", "Unnamed Run")
    display_date = run_data.get("display_date", "Unknown Date")
    scenario = run_data.get("scenario", "N/A")
    feedstock_scenario = run_data.get("feedstock_scenario", "N/A")
    steps = run_data.get("steps", "N/A")
    seed = run_data.get("seed", "N/A")

    results = run_data.get("results_summary", {})
    final_price = results.get("final_saf_price")
    total_supply = results.get("total_supply")
    total_demand = results.get("total_demand")

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(run_name, className="card-title"),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Strong("Date: "),
                                html.Span(display_date),
                                html.Br(),
                                html.Strong("Scenario: "),
                                html.Span(scenario),
                                html.Br(),
                                html.Strong("Feedstock: "),
                                html.Span(feedstock_scenario),
                                html.Br(),
                                html.Strong("Steps: "),
                                html.Span(str(steps)),
                                html.Br(),
                                html.Strong("Seed: "),
                                html.Span(str(seed)),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.H6("Results Summary", className="text-muted"),
                                html.Strong("Final SAF Price: "),
                                html.Span(
                                    f"{final_price:.2f}" if final_price else "N/A"
                                ),
                                html.Br(),
                                html.Strong("Final Supply: "),
                                html.Span(
                                    f"{total_supply:.2f}" if total_supply else "N/A"
                                ),
                                html.Br(),
                                html.Strong("Final Demand: "),
                                html.Span(
                                    f"{total_demand:.2f}" if total_demand else "N/A"
                                ),
                            ],
                            md=6,
                        ),
                    ]
                ),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button(
                                "View Details",
                                id={"type": "view-run-btn", "index": run_id},
                                color="primary",
                                size="sm",
                                className="me-2",
                            ),
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Delete",
                                id={"type": "delete-run-btn", "index": run_id},
                                color="danger",
                                size="sm",
                                outline=True,
                            ),
                        ),
                    ]
                ),
            ]
        ),
        className="mb-3",
        style={
            "backgroundColor": "#f8f9fa",
            "borderRadius": "10px",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.1)",
        },
    )


layout = dbc.Container(
    fluid=True,
    children=[
        html.H2("Run History", className="mb-4"),
        html.P(
            "Browse and manage your saved simulation runs. Each run is automatically saved with its configuration and results.",
            className="text-muted",
        ),
        dcc.Store(id="run-history-refresh-trigger", data=0),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "🔄 Refresh",
                            id="refresh-run-history-btn",
                            color="secondary",
                            size="sm",
                            className="me-2",
                        ),
                        dbc.Button(
                            "🗑️ Clear All",
                            id="clear-all-runs-btn",
                            color="danger",
                            size="sm",
                            outline=True,
                        ),
                    ],
                    className="mb-3",
                )
            ]
        ),
        html.Div(id="run-history-container"),
        # Modal for viewing run details
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Run Details")),
                dbc.ModalBody(id="run-detail-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-run-detail-modal", className="ml-auto")
                ),
            ],
            id="run-detail-modal",
            size="lg",
            is_open=False,
        ),
    ],
)


@callback(
    Output("run-history-container", "children"),
    Output("run-history-refresh-trigger", "data"),
    Input("run-history-refresh-trigger", "data"),
    Input("refresh-run-history-btn", "n_clicks"),
    prevent_initial_call=False,
)
def update_run_history(trigger_data, refresh_clicks):
    """Load and display all saved runs"""
    run_manager = get_run_manager()
    runs = run_manager.get_all_runs()

    if not runs:
        return (
            dbc.Alert(
                "No runs saved yet. Run a simulation from the Model page to save your first run!",
                color="info",
                className="mt-3",
            ),
            trigger_data + 1 if refresh_clicks else trigger_data,
        )

    run_cards = [create_run_card(run) for run in runs]

    return (
        html.Div(run_cards),
        trigger_data + 1 if refresh_clicks else trigger_data,
    )


@callback(
    Output("run-detail-modal", "is_open"),
    Output("run-detail-modal-body", "children"),
    Input({"type": "view-run-btn", "index": dash.ALL}, "n_clicks"),
    Input("close-run-detail-modal", "n_clicks"),
    State("run-detail-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_run_detail_modal(view_clicks, close_clicks, is_open):
    """Show detailed information about a run"""
    from dash import callback_context as ctx

    if not ctx.triggered:
        return is_open, ""

    trigger_id = ctx.triggered[0]["prop_id"]

    # Close button clicked
    if "close-run-detail-modal" in trigger_id:
        return False, ""

    # View button clicked
    if "view-run-btn" in trigger_id:
        # Extract run_id from the trigger
        import json

        trigger_data = json.loads(trigger_id.split(".")[0])
        run_id = trigger_data["index"]

        run_manager = get_run_manager()
        run_data = run_manager.get_run_by_id(run_id)

        if not run_data:
            return (
                True,
                dbc.Alert("Run not found!", color="danger"),
            )

        # Create detailed view
        config = run_data.get("config", {})
        boolean_config = run_data.get("boolean_config", {})

        detail_content = [
            html.H5(f"Run: {run_data.get('run_name', 'Unknown')}"),
            html.Hr(),
            html.H6("Configuration"),
            html.Pre(
                json.dumps(config, indent=2), style={"fontSize": "0.8rem", "maxHeight": "200px", "overflow": "auto"}
            ),
            html.H6("Boolean Configuration"),
            html.Pre(
                json.dumps(boolean_config, indent=2),
                style={"fontSize": "0.8rem", "maxHeight": "200px", "overflow": "auto"},
            ),
        ]

        return True, detail_content

    return is_open, ""


@callback(
    Output("run-history-container", "children", allow_duplicate=True),
    Input({"type": "delete-run-btn", "index": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def delete_run(delete_clicks):
    """Delete a run from the history"""
    from dash import callback_context as ctx
    import json

    if not ctx.triggered or not any(delete_clicks):
        return no_update

    trigger_id = ctx.triggered[0]["prop_id"]

    if "delete-run-btn" in trigger_id:
        # Extract run_id from the trigger
        trigger_data = json.loads(trigger_id.split(".")[0])
        run_id = trigger_data["index"]

        run_manager = get_run_manager()
        run_manager.delete_run(run_id)

        # Reload runs
        runs = run_manager.get_all_runs()

        if not runs:
            return dbc.Alert(
                "No runs saved yet. Run a simulation from the Model page to save your first run!",
                color="info",
                className="mt-3",
            )

        run_cards = [create_run_card(run) for run in runs]
        return html.Div(run_cards)

    return no_update


@callback(
    Output("run-history-container", "children", allow_duplicate=True),
    Input("clear-all-runs-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all_runs(n_clicks):
    """Clear all saved runs"""
    if not n_clicks:
        return no_update

    run_manager = get_run_manager()
    run_manager.clear_all_runs()

    return dbc.Alert(
        "All runs cleared! Run a simulation from the Model page to save a new run.",
        color="warning",
        className="mt-3",
    )
