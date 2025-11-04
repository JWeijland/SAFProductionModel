from __future__ import annotations

 

# Standard library

import json

import ast

 

# Third-party libraries

from flask import app

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import plotly.io as pio

from plotly.subplots import make_subplots

 

# Dash and Dash Bootstrap Components

import dash

from dash import (

    html,

    Input,

    Output,

    State,

    callback_context as ctx,

    no_update,

)

from dash.dependencies import ALL, MATCH

import dash_bootstrap_components as dbc

from dash.exceptions import PreventUpdate

 

# Custom modules

from input_stores import (

    BOOLEAN_CONFIG_KEYS,

    SLIDER_CONFIG_KEYS_SINGLE,

    SLIDER_CONFIG_KEYS_RANGE,

)

from runner import run_market_model_csv, run_market_model_csv_batch

from helper import _parse_list

 

pio.templates.default = "plotly"

 

from helper import _prep_runs, _compute_feed_reliability_per_run, single_keys, range_keys, inputs

 

def register_callbacks(app: dash.Dash) -> None: 

    @app.callback(

        Output("sliders-store", "data"),

        inputs + [Input("btn-reset-defaults", "n_clicks")],

        prevent_initial_call=False

    )

    def update_or_reset_sliders(*args):

        triggered_id = ctx.triggered_id

 

        def _defaults():

            data = {k: SLIDER_CONFIG_KEYS_SINGLE[k]["default"] for k in single_keys}

            data.update({k: SLIDER_CONFIG_KEYS_RANGE[k]["default"] for k in range_keys})

            return data

 

        if triggered_id is None or triggered_id == "btn-reset-defaults":

            store_data = _defaults()

            return store_data

 

        # Otherwise, collect current slider values

        slider_values = args[:-1]  # exclude btn-reset-defaults

 

        all_keys = single_keys + range_keys

        store_payload = {}

        for k, v in zip(all_keys, slider_values):

            # For safety: if a range slider returns None, substitute its default

            if k in SLIDER_CONFIG_KEYS_RANGE and (v is None or not isinstance(v, (list, tuple))):

                store_payload[k] = SLIDER_CONFIG_KEYS_RANGE[k]["default"]

            elif k in SLIDER_CONFIG_KEYS_SINGLE and v is None:

                store_payload[k] = SLIDER_CONFIG_KEYS_SINGLE[k]["default"]

            else:

                store_payload[k] = v

        return store_payload

 

 

    @app.callback(

        Output("run-status", "children"),

        Output("tabs", "value"),

        Output("store-model", "data"),

        Output("store-investor", "data"),

        Output("store-feedstock-aggregator", "data"),

        Output("store-saf-production-site", "data"),

        Output("store-market-metric-log", "data"),

        Output("store-batch-model", "data"),

        Output("store-batch-investor", "data"),

        Output("store-batch-feedstock-aggregator", "data"),

        Output("store-batch-saf-production-site", "data"),

        Output("store-batch-market-metric-log", "data"),

        Input("btn-run", "n_clicks"),

        Input("btn-batch-run", "n_clicks"),

        State("scenario-dd", "value"),

        State("feedstock-scenario-dd", "value"),

        State("steps-slider", "value"),

        State("sliders-store", "data"),

        State("boolean-config-store", "data"),

        State("seed-input", "value"),

        State("runs-input", "value"),

        prevent_initial_call=True,

    )

    def run_simulation(

        n_clicks, n_clicks_batch, scenario, feedstock_scenario, steps, sliders_store, boolean_config_store, seed, runs

    ):

        if not n_clicks and not n_clicks_batch:

            raise PreventUpdate

 

        triggered_id = ctx.triggered_id

 

        if triggered_id == "btn-run":

 

            (

                model_log_df,

                feedstock_aggregator_log_df,

                saf_production_site_log_df,

                investor_log_df,

                market_metric_log,

            ) = run_market_model_csv(

                scenario=scenario or "Surge",

                feedstock_scenario=feedstock_scenario or "Oversupply",

                steps=int(steps or 100),

                config_store=sliders_store,

                boolean_config_store=boolean_config_store,

                seed=seed,

            )

 

            summary = dbc.Alert(

                f"Simulation complete for {scenario or 'Surge'} with {steps or 100} steps.",

                color="success",

                dismissable=True,

                className="mt-3",

            )

 

            return (

                summary,

                "tab-config",

                model_log_df.to_dict("records"),

                investor_log_df.to_dict("records"),

                feedstock_aggregator_log_df.to_dict("records"),

                saf_production_site_log_df.to_dict("records"),

                market_metric_log.to_dict("records"),

                [],

                [],

                [],

                [],

                [],  # Batch outputs

            )

 

        elif triggered_id == "btn-batch-run":

 

            (

                batch_model_df,

                batch_fa_df,

                batch_saf_site_df,

                batch_investor_df,

                batch_market_metrics_df,

            ) = run_market_model_csv_batch(

                scenario=scenario or "Surge",

                feedstock_scenario=feedstock_scenario or "Oversupply",

                steps=int(steps or 100),

                config_store=sliders_store,

                boolean_config_store=boolean_config_store,

                base_seed=seed,

                runs=int(runs or 10),

                prefix="dash_batch",

            )

 

            summary = dbc.Alert(

                f"Batch simulation complete for {scenario or 'Surge'} at {feedstock_scenario or 'Surplus'} with {runs or 10} runs.",

                color="info",

                dismissable=True,

                className="mt-3",

            )

 

            return (

                summary,

                "tab-batch",

                [],

                [],

                [],

                [],

                [],  # single run outputs

                batch_model_df.to_dict("records"),

                batch_investor_df.to_dict("records"),

                batch_fa_df.to_dict("records"),

                batch_saf_site_df.to_dict("records"),

                batch_market_metrics_df.to_dict("records"),

            )

   

    # ---------- Boolean Toggles ----------

    @app.callback(

        Output("boolean-config-store", "data"),

        Input({"type": "boolean-toggle", "index": ALL}, "on"),  # Changed from "on" to "value"

        State({"type": "boolean-toggle", "index": ALL}, "id"),

        prevent_initial_call=False,  # This allows initial population

    )

    def update_boolean_config(toggled_values, id_list):

        """

        Collect current toggle states into a config dict.

        Fires on load (initial values) and on any toggle change.

       

        @param toggled_values: List of boolean values from toggle components

        @param id_list: List of component IDs with structure {"type": "boolean-toggle", "index": key}

        @return: Dictionary mapping config keys to boolean values

        """

        if not id_list:

            # Return defaults if no components exist yet

            return BOOLEAN_CONFIG_KEYS

       

        # If no values yet (very early load), use defaults

        if not toggled_values or len(toggled_values) != len(id_list):

            return BOOLEAN_CONFIG_KEYS

           

        # Build config dict from current toggle states

        config = {}

        for id_obj, val in zip(id_list, toggled_values):

            key = id_obj["index"]

            # Use the toggle value if available, otherwise fall back to default

            config[key] = bool(val) if val is not None else BOOLEAN_CONFIG_KEYS.get(key, False)

       

        return config

 

    @app.callback(

        Output({"type": "boolean-toggle", "index": ALL}, "on"),  # Changed from "on" to "value"

        Input("btn-reset-defaults", "n_clicks"),

        State({"type": "boolean-toggle", "index": ALL}, "id"),

        prevent_initial_call=True,

    )

    def reset_boolean_toggles(n_clicks, id_list):

        """

        Reset all boolean toggles to their default values.

       

        @param n_clicks: Number of times reset button was clicked

        @param id_list: List of toggle component IDs

        @return: List of default boolean values in same order as id_list

        """

        if not n_clicks or not id_list:

            raise PreventUpdate

       

        return [BOOLEAN_CONFIG_KEYS.get(id_obj["index"], False) for id_obj in id_list]

 

    # ----------------- Output Graphs ---------------------------------

 

    # ------------------ Market Metrics ----------------

    @app.callback(

        Output("graph-saf-price-over-time", "figure"),

        Input("store-model", "data"),

    )

    def plot_saf_price_over_time(model_data):

        if not model_data:

            return None

 

        df = pd.DataFrame(model_data)

 

        fig = px.line(df, x="Year", y="Market_Price")

 

        return fig

 

    @app.callback(

        Output("graph-supply-demand", "figure"),

        Input("store-market-metric-log", "data"),

    )

    def plot_supply_demand(data):

        if not data:

            return None

 

        df = pd.DataFrame(data)

 

        x_col = "Year"

        x_label = "Year"

        y_cols = [c for c in ["Total_Supply", "Demand"] if c in df.columns]

        fig = px.line(

            df,

            x=x_col,

            y=y_cols,

            markers=True,

            labels={x_col: x_label, "value": "Volume (tonnes)"},

        )

        fig.update_xaxes(dtick=1)

        fig.update_yaxes(tickformat=",.0f")

        fig.update_layout(template="plotly_white")

 

        return fig

 

    @app.callback(

        Output("consumer-price-forecast-graph", "figure"),

        Input("store-investor", "data"),

    )

    def fig_consumer_price_overlay(data):

        if not data:

            return None

 

        work = pd.DataFrame(data)

 

        x_col = "Year"

        color_col = "Issued_Year"

 

        # Ensure forecast column exists

        if "Consumer_Price_Forecast" not in work.columns:

            return None

 

        # Parse forecast lists

        work["__list"] = work["Consumer_Price_Forecast"].apply(_parse_list)

        work = work.dropna(subset=["__list"])

 

        # Build overlay data

        rows = []

        for _, r in work.iterrows():

            issued = float(r[x_col])

            prices = r["__list"]

            if not prices:

                continue

            for i, val in enumerate(prices):

                if val is None:

                    continue

                rows.append(

                    {

                        x_col: issued + i,

                        "Consumer_Price": float(val),

                        color_col: int(issued),

                    }

                )

 

        overlay = pd.DataFrame(rows)

 

        fig = go.Figure()

 

        for issued_year in sorted(overlay[color_col].unique()):

            group = overlay[overlay[color_col] == issued_year].sort_values(x_col)

            fig.add_trace(

                go.Scatter(

                    x=group[x_col],

                    y=group["Consumer_Price"],

                    mode="lines+markers",

                    name=f"Issued in {issued_year}",

                    hovertemplate=f"Year=%{{x}}<br>Issued in {issued_year}: %{{y:,.0f}}",

                )

            )

 

        fig.update_layout(

            xaxis_title="Year",

            yaxis_title="Consumer Price (USD/tonne)",

            template="plotly_white",

        )

        fig.update_xaxes(dtick=1)

        fig.update_yaxes(tickformat=",.0f")

 

        return fig

 

    # ------------- Production Metrics -----------------

 

    @app.callback(

        Output("graph-supply-by-plant", "figure"),

        Input("store-saf-production-site", "data"),

    )

    def fig_supply_by_plant(data):

        if not data:

            return None

 

        df = pd.DataFrame(data)

 

        x_col = "Year"

        x_label = "Year"

        fig = px.line(

            df,

            x=x_col,

            y="Production_Output",

            color="AgentID",

            markers=True,

            labels={

                x_col: x_label,

                "Production_Output": "Production / Supply (tonnes)",

                "AgentID": "Plant",

            },

        )

        fig.update_traces(line=dict(width=1.5))

        fig.update_xaxes(dtick=1)

        fig.update_yaxes(tickformat=",.0f")

        fig.update_layout(template="plotly_white")

 

        return fig

 

    @app.callback(

        Output("state-dropdown", "options"),

        Output("state-dropdown", "value"),

        Input("store-feedstock-aggregator", "data"),

    )

    def populate_state_dropdown(data):

        if not data:

            return [], None

 

        df = pd.DataFrame(data)

        unique_states = sorted(df["State_ID"].dropna().unique())

 

        options = [{"label": "Combined", "value": "ALL"}] + [

            {"label": state, "value": state} for state in unique_states

        ]

 

        default = unique_states[:3] if unique_states else []

 

        return options, default

 

    @app.callback(

        Output("feedstock-availability-graph", "figure"),

        Input("state-dropdown", "value"),

        Input("store-feedstock-aggregator", "data"),

        prevent_initial_call=True,

    )

    def update_feedstock_graph(selected_states, data):

        if not data or not selected_states:

            return None

 

        if isinstance(selected_states, str):

            selected_states = [selected_states]

 

        df = pd.DataFrame(data)

 

        # Ensure numeric types

        df["Year"] = pd.to_numeric(df.get("Year"), errors="coerce")

        if "Available_Feedstock" not in df.columns:

            df["Available_Feedstock"] = 0

        if "Current_Supply" not in df.columns:

            df["Current_Supply"] = 0

 

        df["Available_Feedstock"] = pd.to_numeric(

            df["Available_Feedstock"], errors="coerce"

        ).fillna(0)

        df["Current_Supply"] = pd.to_numeric(

            df["Current_Supply"], errors="coerce"

        ).fillna(0)

 

        # Filter states if not ALL

        if "ALL" not in selected_states:

            df = df[df["State_ID"].isin(selected_states)]

 

        # Group by Year and aggregate both series

        grp = (

            df.groupby("Year", as_index=False)[

                ["Available_Feedstock", "Current_Supply"]

            ]

            .sum()

            .sort_values("Year")

        )

 

        # Remove the first year (initiation year)

        if not grp.empty:

            first_year = grp["Year"].min()

            grp = grp[grp["Year"] != first_year]

 

        # Build overlay bars: Current Supply behind, Available Feedstock in front

        fig = go.Figure()

 

        # Behind (will only be visible above the front bar if larger)

        fig.add_bar(

            x=grp["Year"],

            y=grp["Current_Supply"],

            name="Current Supply",

            marker_color="#ff6600",  # pale blue (matches your UI preference)

            opacity=0.95,  # back bar; front is fully opaque

            hovertemplate="%{x}<br>Current Supply: %{y:,.0f} tonnes<extra></extra>",

        )

 

        # Front (fully opaque so it masks the back except where the back is taller)

        fig.add_bar(

            x=grp["Year"],

            y=grp["Available_Feedstock"],

            name="Available Feedstock",

            marker_color="#2aa876",  # contrasting green

            opacity=1.0,

            hovertemplate="%{x}<br>Available Feedstock: %{y:,.0f} tonnes<extra></extra>",

        )

 

        fig.update_layout(

            barmode="overlay",  # <-- overlay (not stack)

            template="plotly_white",

            hovermode="x unified",

            legend_title_text="Legend",

            bargap=0.15,  # keep bars exactly over each other

            bargroupgap=0,

        )

 

        fig.update_xaxes(dtick=1, title_text="Year")

        fig.update_yaxes(tickformat=",.0f", title_text="Tonnes")

 

        return fig

 

    # ------------- Econometric Metrics ---------------

 

    @app.callback(

        Output("graph-owned-sites-per-investor", "figure"),

        Input("store-investor", "data"),

    )

    def fig_owned_sites_per_investor(data):

        if not data:

            return None

 

        df = pd.DataFrame(data)

        latest_tick_df = df.sort_values("Tick").groupby("Investor_ID").tail(1)

        site_distribution = (

            latest_tick_df["Num_Owned_Sites"].value_counts().sort_index().reset_index()

        )

        site_distribution.columns = ["Number of Sites Owned", "Number of Investors"]

 

        fig = px.bar(

            site_distribution,

            x="Number of Sites Owned",

            y="Number of Investors",

            labels={

                "Number of Sites Owned": "Number of Sites Owned",

                "Number of Investors": "Number of Investors",

            },

        )

        return fig

 

    @app.callback(

        Output("investor-kpis-graph", "figure"),

        Input("investor-metrics-dropdown", "value"),

        Input("store-investor", "data"),

        prevent_initial_call=True,

    )

    def fig_investor_kpis(investor_id, data):

        if not data:

            return None

 

        df = pd.DataFrame(data)

        df["Investor_ID"] = df["Investor_ID"].astype(str)

        df = df[df["Investor_ID"] == str(investor_id)].copy()

 

        x_col = "Year"

        x_label = "Year"

        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")

 

        has_ebit = "EBIT" in df.columns

        has_dr = "Discount_Rate" in df.columns

        has_roace = "Raw_ROACE" in df.columns

 

        if "Num_Owned_Sites" in df.columns:

            df["Num_Owned_Sites"] = pd.to_numeric(df["Num_Owned_Sites"], errors="coerce")

        if has_ebit:

            df["EBIT"] = pd.to_numeric(df["EBIT"], errors="coerce")

 

        keep = [x_col] + [c for c in ["EBIT", "Discount_Rate", "Raw_ROACE", "Num_Owned_Sites"] if c in df.columns]

        df = df[keep].dropna(subset=[x_col]).sort_values(x_col)

 

        # --- For Sites trace only: inject initial (min_year - 1) = 0 sites to show a baseline ---

        min_year = df[x_col].min() if not df.empty else None

        initial_year = int(min_year) - 1 if pd.notna(min_year) else None

 

        # --- Prepare figure ---

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        c_ebit, c_dr, c_roace = "#3A8CC7", "#E07A5F", "#2A9D8F"

        # EBIT

        if has_ebit:

            fig.add_trace(

                go.Scatter(

                    x=df[x_col],

                    y=df["EBIT"],

                    name="EBIT (USD)",

                    marker_color=c_ebit,

                    hovertemplate=f"{x_label}=%{{x}}&lt;br&gt;EBIT=%{{y:,.0f}}&lt;extra&gt;&lt;/extra&gt;",

                ),

                secondary_y=False,

            )

 

        # Discount Rate

        if has_dr and "Discount_Rate" in df.columns and not df["Discount_Rate"].dropna().empty:

            fig.add_trace(

                go.Scatter(

                    x=df[x_col],

                    y=df["Discount_Rate"],

                    name="Discount Rate",

                    mode="lines+markers",

                    line=dict(color=c_dr, width=2),

                    hovertemplate=f"{x_label}=%{{x}}&lt;br&gt;DR=%{{y:.2%}}&lt;extra&gt;&lt;/extra&gt;",

                ),

                secondary_y=True,

            )

 

        # ROACE

        if has_roace:

            fig.add_trace(

                go.Scatter(

                    x=df[x_col],

                    y=df["Raw_ROACE"],

                    name="ROACE",

                    mode="lines+markers",

                    line=dict(color=c_roace, width=2, dash="dot"),

                    hovertemplate=f"{x_label}=%{{x}}&lt;br&gt;ROACE=%{{y:.2%}}&lt;extra&gt;&lt;/extra&gt;",

                ),

                secondary_y=True,

            )

 

        sites_base = None

        if "Num_Owned_Sites" in df.columns:

            sites_base = (

                df[[x_col, "Num_Owned_Sites"]]

                .dropna(subset=[x_col])

                .assign(**{x_col: lambda d: d[x_col].astype(int)})

            )

            sites_base = (

                sites_base.groupby(x_col, as_index=True)["Num_Owned_Sites"]

                .max()

                .sort_index()

            )

        if "Num_Owned_Sites" in df.columns:

            sites_df = df[[x_col, "Num_Owned_Sites"]].copy().dropna(subset=[x_col])

            if initial_year is not None:

                sites_df = pd.concat(

                    [sites_df, pd.DataFrame({x_col: [initial_year], "Num_Owned_Sites": [0]})],

                    ignore_index=True,

                )

            if not sites_df.empty:

                try:

                    sites_df[x_col] = sites_df[x_col].astype(int)

                except Exception:

                    pass

                sites_df = sites_df.sort_values(x_col)

 

                fig.add_trace(

                    go.Scatter(

                        x=sites_df[x_col],

                        y=sites_df["Num_Owned_Sites"],

                        name="Sites Owned",

                        mode="lines",

                        line=dict(color="#F4A261", width=2),

                        yaxis="y3",

                        hovertemplate=f"{x_label}=%{{x}}&lt;br&gt;Sites=%{{y}}&lt;extra&gt;&lt;/extra&gt;",

                    )

                )

 

        if sites_base is not None and len(sites_base) >= 2:

            s = sites_base.copy()

            increase_years = s.index[s.diff() > 0].tolist() 

 

            if increase_years:

                data_year_min = int(s.index.min())

                data_year_max = int(s.index.max())

 

                # Build [start, end] as numeric axis spans covering full year blocks.

                # "Afterwards" => shade years Y+1 .. Y+4 inclusive.

                # Use -0.5/+0.5 padding so each integer year is fully shaded visually.

                spans = []

                for y in increase_years:

                    start = (y + 1) - 0.5

                    end = (y + 4) + 0.5

                    # Clamp to data domain (optional, keeps shading within visible range)

                    start = max(start, data_year_min - 0.5)

                    end = min(end, data_year_max + 0.5)

                    if start < end:

                        spans.append((start, end))

 

                # Merge overlapping/adjacent spans

                spans.sort(key=lambda t: t[0])

                merged = []

                for st, en in spans:

                    if not merged or st > merged[-1][1]:

                        merged.append([st, en])

                    else:

                        merged[-1][1] = max(merged[-1][1], en)

 

                # Add the vrects behind traces

                for st, en in merged:

                    fig.add_vrect(

                        x0=st,

                        x1=en,

                        fillcolor="lightgray",

                        opacity=0.25,

                        layer="below",

                        line_width=0,

                    )

 

                # (Optional) Legend proxy for shaded areas — uncomment to show a legend item

                # fig.add_trace(

                #     go.Scatter(

                #         x=[None], y=[None],

                #         mode="markers",

                #         marker=dict(size=0, color="lightgray"),

                #         name="Post-Increase Window (4 yrs)",

                #         showlegend=True,

                #     )

                # )

 

        fig.update_xaxes(title_text=x_label, dtick=1)

        fig.update_yaxes(title_text="EBIT (USD)", secondary_y=False, tickformat=",.0f")

        fig.update_yaxes(title_text="Rates (%)", secondary_y=True, tickformat=".1%")

        fig.update_layout(

            hovermode="x unified",

            legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),

            margin=dict(l=40, r=40, t=60, b=40),

            template="plotly_white",

        )

        fig.update_layout(

            yaxis3=dict(

                title="Assets Owned",

                overlaying="y",

                side="right",

                position=1.0,

                tickfont=dict(color="#F4A261"),

            )

        )

 

        return fig

   

    @app.callback(

        Output("investor-metrics-dropdown", "options"),

        Output("investor-metrics-dropdown", "value"),

        Input("store-investor", "data"),

    )

    def populate_investor_dropdown(data):

        if not data:

            return [], None

        df = pd.DataFrame(data)

        unique_investors = sorted(df["Investor_ID"].dropna().unique())

        options = [{"label": inv, "value": str(inv)} for inv in unique_investors]

        default = unique_investors[0] if unique_investors else None

        return options, default

 

    @app.callback(

        Output("plant-ebit-graph", "figure"),

        Input("store-saf-production-site", "data"),

        Input("dropdown-selected-plants", "value"),

        prevent_initial_call=True,

    )

    def fig_plant_ebit_line(site_df: pd.DataFrame, selected_plants) -> go.Figure:

        if not site_df or not selected_plants:

            return None

        df = pd.DataFrame(site_df)

        selected_plants = [str(p) for p in selected_plants if p is not None]

 

        x_col = "Year" if "Year" in df.columns else "Tick"

        x_label = "Year" if x_col == "Year" else "Tick"

 

        filtered_df = df[df["AgentID"].isin(selected_plants)]

 

        fig = px.line(

            filtered_df,

            x=x_col,

            y="EBIT",

            color="AgentID",

            markers=True,

            labels={x_col: x_label, "EBIT": "EBIT (USD)", "AgentID": "Plant"},

        )

        fig.update_traces(

            line=dict(width=2),

            hovertemplate=f"{x_label}=%{{x}}<br>%{{fullData.name}} EBIT=%{{y:,.0f}}<extra></extra>",

        )

        fig.update_xaxes(dtick=1)

        fig.update_yaxes(tickformat=",.0f", title_text="EBIT (USD)")

        fig.update_layout(template="plotly_white", legend_title_text="Plant")

        return fig

 

    @app.callback(

        Output("dropdown-selected-plants", "options"),

        Output("dropdown-selected-plants", "value"),

        Input("store-saf-production-site", "data"),

    )

    def update_dropdown_options(site_data):

        if not site_data:

            return [], None

 

        site_df = pd.DataFrame(site_data)

 

        if "AgentID" not in site_df.columns:

            return [], None

 

        unique_plants = site_df["AgentID"].dropna().unique()

        dropdown_options = [

            {"label": agent_id, "value": agent_id} for agent_id in unique_plants

        ]

 

        # No default selection

        return dropdown_options, None

 

    @app.callback(

        Output("plant-graph", "figure"),

        Input("plant-dropdown", "value"),

        State("store-saf-production-site", "data"),

    )

    def update_plant_graph(selected_plants, site_data):

        if not site_data or not selected_plants:

            return go.Figure()

 

        site_df = pd.DataFrame(site_data)

        return fig_plant_ebit_line(site_df, selected_plants)

 

    @app.callback(

        Output("feedstock-collapse", "is_open"),

        Output("feedstock-toggle", "className"),

        Input("feedstock-toggle", "n_clicks"),

        State("feedstock-collapse", "is_open"),

        prevent_initial_call=True,

    )

    def toggle_feedstock_panel(n_clicks, is_open):

        open_now = not (is_open or False)

        btn_class = (

            "feedstock-toggle-btn is-open" if open_now else "feedstock-toggle-btn"

        )

        return open_now, btn_class

 

    @app.callback(

        Output("feedstock-list", "children"),

        Input("store-feedstock-aggregator", "data"),

    )

    def update_feedstock_info(data):

        if not data:

            return [

                dbc.ListGroupItem(

                    "No feedstock prices available yet.",

                    class_name="feedstock-item empty",

                )

            ]

 

        df = pd.DataFrame(data)

        df = df.dropna(subset=["AgentID", "Feedstock_Price"]).copy()

        df["Feedstock_Price"] = pd.to_numeric(df["Feedstock_Price"], errors="coerce")

        df = df.dropna(subset=["Feedstock_Price"])

        if df.empty:

            return [

                dbc.ListGroupItem(

                    "No valid feedstock prices available.",

                    class_name="feedstock-item empty",

                )

            ]

 

        df = df.drop_duplicates(subset=["AgentID"])

        df = df.sort_values("Feedstock_Price")

 

        min_cost = df["Feedstock_Price"].min()

 

        items = []

        for _, row in df.iterrows():

            is_lowest = row["Feedstock_Price"] == min_cost

            price_str = f"${row['Feedstock_Price']:,.2f}/t"

 

            badge = (

                dbc.Badge(

                    "Lowest", color="info", pill=True, className="ms-2 badge-lowest"

                )

                if is_lowest

                else None

            )

 

            items.append(

                dbc.ListGroupItem(

                    dbc.Row(

                        [

                            dbc.Col(

                                html.Span(row["AgentID"], className="agent-name"),

                                width=7,

                            ),

                            dbc.Col(

                                html.Span(

                                    [price_str, " ", badge] if badge else price_str,

                                    className="price",

                                ),

                                width=5,

                                className="text-end",

                            ),

                        ],

                        className="g-0 align-items-center",

                    ),

                    class_name=f"feedstock-item{' lowest' if is_lowest else ''}",

                )

            )

 

        return items

 

    @app.callback(

        Output("fs-modal-graph", "figure"),

        Input("fs-modal-fig-store", "data"),

        prevent_initial_call=True,

    )

    def update_modal_graph(fig_json):

        if fig_json is None:

            return no_update

 

        # Convert JSON string to Python dict if needed

        return json.loads(fig_json) if isinstance(fig_json, str) else fig_json

 

    @app.callback(

        Output("graph-production-by-investor-vs-demand", "figure"),

        Input("store-saf-production-site", "data"),

    )

    def plot_production_by_investor_vs_demand(investor_data):

        # Return an empty figure if there is no data yet

        if not investor_data:

            return go.Figure()

 

        df = pd.DataFrame(investor_data)

 

        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

        df["Production_Output"] = pd.to_numeric(df["Production_Output"], errors="coerce").fillna(0.0)

        df["Investor_ID"] = df["Investor_ID"].fillna("Unknown")

 

        # If SAF_Demand exists, coerce it so we can optionally plot it

        has_demand = "SAF_Demand" in df.columns

        if has_demand:

            df["SAF_Demand"] = pd.to_numeric(df["SAF_Demand"], errors="coerce")

 

        # Drop rows without a valid year

        df = df.dropna(subset=["Year"]).copy()

        df["Year"] = df["Year"].astype(int)

 

        # --- Aggregate production: Year × Investor_ID ---

        prod = (

            df.groupby(["Year", "Investor_ID"], as_index=False)["Production_Output"]

            .sum()

            .sort_values(["Year", "Investor_ID"])

        )

 

        # Order investors by total production across all years for a stable legend/stack order

        investor_order = (

            prod.groupby("Investor_ID")["Production_Output"]

            .sum()

            .sort_values(ascending=False)

            .index.tolist()

        )

 

        # --- Build stacked bars ---

        fig = px.bar(

            prod,

            x="Year",

            y="Production_Output",

            color="Investor_ID",

            barmode="stack",

            category_orders={"Investor_ID": investor_order},

            labels={

                "Year": "Year",

                "Production_Output": "Production Output",

                "Investor_ID": "Investor"

            },

        )

 

        # --- Optional: overlay annual demand as a line (same y-axis, same units) ---

        if has_demand:

            # If multiple rows per year have SAF_Demand, use the max non-null to avoid double counting

            demand = (

                df.groupby("Year", as_index=False)["SAF_Demand"]

                .max(numeric_only=True)

                .dropna(subset=["SAF_Demand"])

                .sort_values("Year")

            )

 

            if not demand.empty:

                fig.add_scatter(

                    x=demand["Year"],

                    y=demand["SAF_Demand"],

                    mode="lines+markers",

                    name="Demand",

                    line=dict(color="black", width=2),

                    marker=dict(size=6),

                )

 

        # --- Aesthetics ---

        fig.update_layout(

            title="SAF Production by Investor (Stacked) vs Demand",

            legend_title="Investor",

            bargap=0.15,

            xaxis=dict(dtick=1, tickmode="linear"),

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)",

            hovermode="x unified",

        )

        fig.update_traces(

            hovertemplate="<b>Year %{x}</b><br>%{fullData.name}: %{y:,.0f}<extra></extra>"

        )

 

        return fig

 

       

        

    

    

    

    

    

    

    

    

    # ----------------- Batch Graphs -----------------

 

 

    # ----------------- Production Metrics -----------------

 

    @app.callback(

        Output("graph-supply-minus-demand", "figure"),

        Input("store-batch-model", "data"),

    )

    def plot_supply_minus_demand_band(batch_data):

        # Handle empty / missing input

        if not batch_data:

            return None

       

        df = pd.DataFrame(batch_data)

 

        # Validate required columns

        required_cols = {"Year", "Total_Supply", "Demand"}

        if not required_cols.issubset(df.columns):

            missing = ", ".join(sorted(required_cols - set(df.columns)))

            return None

 

        # Ensure numerics and drop rows without needed values

        for col in ["Year", "Total_Supply", "Demand"]:

            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Year", "Total_Supply", "Demand"])

 

        if df.empty:

            return None

 

        # Compute gap, then aggregate by Year across runs

        df["Supply_Minus_Demand"] = df["Total_Supply"] - df["Demand"]

 

        agg = (

            df.groupby("Year")["Supply_Minus_Demand"]

            .agg(

                mean="mean",

                p10=lambda s: s.quantile(0.10, interpolation="linear"),

                p90=lambda s: s.quantile(0.90, interpolation="linear"),

            )

            .reset_index()

            .sort_values("Year")

        )

 

 

        x = agg["Year"]

        p10 = agg["p10"]

        p90 = agg["p90"]

        mean = agg["mean"]

 

        fig = go.Figure()

 

        # --- Shaded band between P10 and P90 ---

        # 1) Lower boundary (no legend entry, invisible line)

        fig.add_trace(

            go.Scatter(

                x=x,

                y=p10,

                mode="lines",

                line=dict(width=0),

                name="P10–P90 band",

                hovertemplate="Year=%{x}<br>P10=%{y:,.0f}<extra></extra>",

                showlegend=False,

            )

        )

        # 2) Upper boundary with fill to previous trace (creates the band)

        fig.add_trace(

            go.Scatter(

                x=x,

                y=p90,

                mode="lines",

                line=dict(width=0),

                fill="tonexty",

                fillcolor="rgba(158, 202, 225, 0.35)",  # pale blue shade

                name="P10–P90",

                hovertemplate="Year=%{x}<br>P90=%{y:,.0f}<extra></extra>",

                showlegend=True,

            )

        )

 

        # --- Mean line ---

        fig.add_trace(

            go.Scatter(

                x=x,

                y=mean,

                mode="lines",

                line=dict(color="#1f77b4", width=3),  # darker blue

                name="Mean",

                hovertemplate="Year=%{x}<br>Mean=%{y:,.0f}<extra></extra>",

                showlegend=True,

            )

        )

 

        # Layout & styling

        fig.update_layout(

 

            xaxis_title="Year",

            yaxis_title="Supply − Demand (tonnes)",

            template="plotly_white",

            hovermode="x unified",

            legend_title="Statistic",

        )

        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="#888")

 

        return fig

 

    @app.callback(

        Output("graph-feed-reliability-heatmap", "figure"),

        Input("store-batch-model", "data"),

    )

    def reliability_heatmap(model_data):

 

        if not model_data:

            return None

 

        df = pd.DataFrame(model_data)

 

        # Ensure 'run' indexing etc.

        df = _prep_runs(df)

 

        # Coerce numerics and drop incomplete rows

        for col in ["Year", "Total_Supply", "Demand"]:

            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["run", "Year", "Total_Supply", "Demand"])

 

        # Aggregate to (run, Year) totals to avoid double counting

        gy = (

            df.groupby(["run", "Year"], as_index=False)

            .agg(Total_Supply=("Total_Supply", "sum"),

                Demand=("Demand", "sum"))

        )

 

        # Classify vs demand with ±10% bands

        # class = -1 (red)  if Supply <  90% of Demand

        # class =  0 (green) if 90% <= Supply <= 110% of Demand

        # class = +1 (blue) if Supply > 110% of Demand

        def classify(row):

            supply = row["Total_Supply"]

            demand = row["Demand"]

            if demand <= 0:

                # If demand is zero, only "within" if supply is also zero, else "above"

                if supply <= 0:

                    return 0, 0.0

                else:

                    return 1, float("inf")

            ratio = supply / demand

            diff_pct = (supply - demand) / demand * 100.0

            if ratio < 0.9:

                return -1, diff_pct

            elif ratio <= 1.1:

                return 0, diff_pct

            else:

                return 1, diff_pct

 

        cls, diff = zip(*gy.apply(classify, axis=1))

        gy["class"] = cls        # -1, 0, +1

        gy["diff_pct"] = diff    # signed % difference

 

        # Pivot to matrices for imshow

        mat_vals = (

            gy.pivot(index="run", columns="Year", values="class")

            .sort_index()

            .sort_index(axis=1)

        )

        mat_diff = (

            gy.pivot(index="run", columns="Year", values="diff_pct")

            .reindex(index=mat_vals.index, columns=mat_vals.columns)

        )

 

        # Build human-friendly hover text

        val_arr = mat_vals.values

        diff_arr = mat_diff.values

        text = np.empty_like(val_arr, dtype=object)

 

        for i in range(val_arr.shape[0]):

            for j in range(val_arr.shape[1]):

                v = val_arr[i, j]

                if pd.isna(v):

                    text[i, j] = "No data"

                    continue

                label = (

                    "Supply < 90% of demand" if v == -1

                    else "Within ±10% of demand" if v == 0

                    else "Supply > 110% of demand"

                )

                d = diff_arr[i, j]

                suffix = f" ({d:+.1f}%)" if pd.notna(d) and np.isfinite(d) else " (N/A)"

                text[i, j] = label + suffix

 

        # Three-color scale mapped to -1, 0, +1

        fig = px.imshow(

            mat_vals,

            aspect="auto",

            origin="lower",

            zmin=-1, zmax=1,

            color_continuous_scale=[

                (0.0, "#e41a1c"),  # red

                (0.5, "#00ff22"),  # green

                (1.0, "#377eb8"),  # blue

            ],

            labels=dict(color="Supply vs demand")

        )

 

        fig.update_layout(

            template="plotly_white",

            coloraxis_colorbar=dict(

                title="Classification",

                tickmode="array",

                tickvals=[-1, 0, 1],

                ticktext=["< -10%", "±10%", "> +10%"],

            )

        )

        fig.update_traces(

            text=text,

            hovertemplate="Run %{y}<br>Year %{x}<br>%{text}<extra></extra>"

        )

 

        return fig

 

    @app.callback(

    Output("kpi-feed-reliability", "figure"),

    Input("store-batch-model", "data"),

    )

    def kpi_feed_reliability(model_data):

        import numpy as np

        import pandas as pd

        import plotly.graph_objects as go

 

        if not model_data:

            return None

 

        df = pd.DataFrame(model_data)

 

        # Ensure 'run' etc.

        df = _prep_runs(df)

 

        # Coerce numerics and drop incomplete rows

        for col in ["Year", "Total_Supply", "Demand"]:

            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["run", "Year", "Total_Supply", "Demand"])

 

        # Aggregate to (run, Year) totals

        gy = (

            df.groupby(["run", "Year"], as_index=False)

            .agg(Total_Supply=("Total_Supply", "sum"),

                Demand=("Demand", "sum"))

        )

 

        # Reliability (lower bound only): YES if Supply ≥ 90% of Demand

        # When Demand <= 0: count YES only if Supply <= 0 (to avoid false positives)

        def meets_90pct(row):

            supply = row["Total_Supply"]

            demand = row["Demand"]

            if demand <= 0:

                return int(supply <= 0)

            return int(supply >= 0.9 * demand)

 

        gy["reliable"] = gy.apply(meets_90pct, axis=1)

 

        # Per-run reliability = fraction of years that are reliable

        per_run = gy.groupby("run", as_index=False)["reliable"].mean()

 

        # Aggregate across runs for the gauge

        mean_val = float(per_run["reliable"].mean()) if not per_run.empty else 0.0

 

        fig = go.Figure(go.Indicator(

            mode="gauge+number",

            value=100.0 * mean_val,

            number={"suffix": "%", "font": {"size": 40}},

            gauge={

                "axis": {"range": [0, 100]},

                "bar": {"color": "#1f77b4"},

                "bgcolor": "rgba(0,0,0,0)",

                "borderwidth": 0,

                "steps": [

                    {"range": [0, 60], "color": "rgba(220, 53, 69, 0.15)"},   # red-ish

                    {"range": [60, 80], "color": "rgba(255, 193, 7, 0.15)"},   # amber

                    {"range": [80, 100], "color": "rgba(40, 167, 69, 0.15)"},  # green

                ],

                "threshold": {"line": {"color": "#28a745", "width": 2}, "thickness": 0.75, "value": 80}

            },

        ))

        fig.update_layout(

            title="Feed Reliability (Supply ≥ 90% of Demand)",

            template="plotly_white",

            margin=dict(l=10, r=10, t=60, b=10)

        )

 

        return fig

 

 

 

    # ----------------- Investor Metrics -----------------

 

    @app.callback(

        Output("graph-investors-at-endyear-hist", "figure"),

        Input("store-batch-investor", "data"),

    )

    def plot_investors_at_endyear_hist(investor_data):

        if not investor_data:

            return None

 

        df = pd.DataFrame(investor_data)

        id_col = "AgentID"

 

        required_cols = {"Year", "run"}

        # If seed exists, we use it to uniquely identify runs

        if "seed" in df.columns:

            required_cols.add("seed")

        # We require an ID column to count unique investors

        if id_col:

            required_cols.add(id_col)

 

        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        df["run"] = pd.to_numeric(df["run"], errors="coerce")

        if "seed" in df.columns:

            df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

 

        subset_cols = ["Year", "run"] + (["seed"] if "seed" in df.columns else [])

        df = df.dropna(subset=subset_cols)

 

        if id_col:

            df[id_col] = df[id_col].astype(str)

            df = df[df[id_col].str.len() > 0]

 

 

        if "seed" in df.columns:

            df["run"] = (

                df["run"].astype("Int64").astype(str) + "|" + df["seed"].astype("Int64").astype(str)

            )

        else:

            df["run"] = df["run"].astype("Int64").astype(str)

 

        df["end_year_of_run"] = df.groupby("run")["Year"].transform("max")

        df_end = df[df["Year"] == df["end_year_of_run"]].copy()

 

        counts = (

            df_end.groupby("run")[id_col]

                .nunique(dropna=True)

                .reset_index(name="investor_count")

        )

 

        start = int(counts["investor_count"].min())

        end = int(counts["investor_count"].max())

        fig = px.histogram(

            counts,

            x="investor_count",

            nbins=max(1, end - start + 1),

        )

 

        fig.update_traces(marker_color="#1f77b4", opacity=0.9)

        fig.update_layout(

            xaxis_title="Number of Investors at End Year (per run)",

            yaxis_title="Number of Runs",

            template="plotly_white",

            bargap=0.1,

        )

        fig.update_xaxes(dtick=1) 

 

        return fig

 

    @app.callback(

    Output("graph-max-sites-owned", "figure"),

    Input("store-batch-investor", "data"),

    )

    def plot_max_sites_owned(investor_data):

        if not investor_data:

            return None

 

 

        df = pd.DataFrame(investor_data)

 

        # Ensure numeric types

        df["Num_Owned_Sites"] = pd.to_numeric(df["Num_Owned_Sites"], errors="coerce")

        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        df["run"] = pd.to_numeric(df["run"], errors="coerce")

 

 

        # Get final year per run

        final_years = df.groupby("run")["Year"].max().reset_index()

        final_df = pd.merge(df, final_years, on=["run", "Year"])

 

        # Get max sites owned at final year per run

        max_sites_final_year = final_df.groupby("run")["Num_Owned_Sites"].max().reset_index()

 

        # Count occurrences of each max site value

        site_counts = max_sites_final_year["Num_Owned_Sites"].value_counts().reset_index()

        site_counts.columns = ["Num_Owned_Sites", "Count"]

        site_counts = site_counts.sort_values("Num_Owned_Sites")

 

        # Create bar chart

        fig = px.bar(

            site_counts,

            x="Num_Owned_Sites",

            y="Count",

            labels={"Num_Owned_Sites": "Max Sites Owned", "Count": "Number of Runs"}

        )

 

        return fig

   

    @app.callback(

        Output("graph-avg-sites-distribution", "figure"),

        Input("store-batch-investor", "data"),

    )

    def plot_avg_sites_distribution(investor_data):

        if not investor_data:

            return None

 

        df = pd.DataFrame(investor_data)

 

        # Group by run and AgentID to get final owned sites per investor per run

        investor_sites = df.groupby(["run", "AgentID"])["Num_Owned_Sites"].max().reset_index()

 

        # Compute average sites per investor per run

        avg_sites_per_run = investor_sites.groupby("run")["Num_Owned_Sites"].mean().reset_index()

        avg_sites_per_run.columns = ["run", "Avg_Sites_Per_Investor"]

 

        # Create histogram

        fig = px.histogram(

            avg_sites_per_run,

            x="Avg_Sites_Per_Investor",

            nbins=20,

            labels={"Avg_Sites_Per_Investor": "Average Sites per Investor", "count": "Number of Runs"}

        )

 

        return fig

 

    @app.callback(

        Output("graph-plants-per-investor-vs-investors", "figure"),

        Input("store-batch-model", "data"),

    )

    def scatter_plants_per_investor_vs_investors(batch_data):

        # Handle empty input

        if not batch_data:

            return None

 

        df = pd.DataFrame(batch_data)

 

        # Ensure runs are stamped (use your helper if available)

        prep = globals().get("_prep_runs", None)

        if callable(prep):

            df = prep(df)

        else:

            if "run" not in df.columns:

                df["run"] = 0  # fallback single-run label

 

        # Validate columns

        required = {"Num_Investors", "Num_Production_Sites"}

        has_tick = "Tick" in df.columns

        has_year = "Year" in df.columns

        if not required.issubset(df.columns) or not (has_tick or has_year):

            fig = go.Figure()

            fig.add_annotation(

                text="Missing columns: need Num_Investors, Num_Production_Sites and Tick or Year",

                showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",

                font={"color": "#5b6b80"},

            )

            return fig

 

        # Coerce numerics

        for col in ["Num_Investors", "Num_Production_Sites"]:

            df[col] = pd.to_numeric(df[col], errors="coerce")

        if has_tick:

            df["Tick"] = pd.to_numeric(df["Tick"], errors="coerce")

        if has_year:

            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

 

        # Drop rows missing keys

        drop_subset = ["run", "Num_Investors", "Num_Production_Sites"]

        drop_subset.append("Tick" if has_tick else "Year")

        df = df.dropna(subset=drop_subset)

        if df.empty:

            return go.Figure()

 

        # End-of-run row: max Tick (preferred) else max Year per run

        if has_tick and df["Tick"].notna().any():

            idx_last = df.groupby("run")["Tick"].idxmax()

        else:

            idx_last = df.groupby("run")["Year"].idxmax()

        end = df.loc[idx_last].copy()

 

        # Compute average plants per investor; exclude runs with zero investors (undefined)

        end["Avg_Plants_per_Investor"] = np.where(

            end["Num_Investors"] > 0,

            end["Num_Production_Sites"] / end["Num_Investors"],

            np.nan

        )

        end = end.dropna(subset=["Avg_Plants_per_Investor"])

        if end.empty:

            fig = go.Figure()

            fig.add_annotation(

                text="No valid runs at end-of-run (Num_Investors == 0)",

                showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",

                font={"color": "#5b6b80"},

            )

            return fig

 

        # Optional: Make run labels tidy strings for hover/text

        end["run"] = end["run"].astype(str)

 

        # Build scatter plot

        fig = px.scatter(

            end,

            x="Num_Investors",

            y="Avg_Plants_per_Investor",

            hover_data={

                "run": True,

                "Num_Investors": ":,.0f",

                "Num_Production_Sites": ":,.0f",

                "Avg_Plants_per_Investor": ":.2f",

            },

            labels={

                "Num_Investors": "Total Investors (end of run)",

                "Avg_Plants_per_Investor": "Avg Plants per Investor (end of run)",

            },

        )

 

        # Style

        fig.update_traces(marker=dict(size=10, color="#1f77b4", opacity=0.9))

        fig.update_layout(

            template="plotly_white",

            hovermode="closest",

            xaxis=dict(tickmode="linear", dtick=1, zeroline=True, zerolinecolor="#B0C4DE"),

            yaxis=dict(zeroline=True, zerolinecolor="#B0C4DE"),

            margin=dict(l=10, r=10, t=60, b=10),

        )

 

        return fig

 

 

    @app.callback(

        Output("graph-optimism-vs-share", "figure"),

        Input("store-batch-investor", "data"),  

        Input("store-batch-saf-production-site", "data"),  

    )

    def plot_optimism_vs_market_share(inv_data, prod_data):

        # Defensive checks

        if not inv_data or not prod_data:

            return None

 

        inv_df = pd.DataFrame(inv_data)

        prod_df = pd.DataFrame(prod_data)

 

 

 

        # Harmonize IDs

        inv_df = inv_df.rename(columns={"AgentID": "Investor_ID"})

 

        # --- Get one Optimism_Factor per (run, seed, Investor_ID) ---

        # If Optimism_Factor repeats per year, take the first non-null

        optim = (inv_df

                .sort_values(["run", "seed", "Investor_ID", "Year"])

                .groupby(["run", "seed", "Investor_ID"], as_index=False)

                .agg(Optimism_Factor=("Optimism_Factor", lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)))

 

                # Keep only the final year per (run, seed)

        last_year = prod_df.groupby(["run", "seed"], as_index=False)["Year"].max()

        prod_df = prod_df.merge(last_year, on=["run", "seed", "Year"], how="inner")

 

 

        # --- Aggregate production to (run, seed, Investor_ID), sum across all years in that run/seed ---

        prod_df["Production_Output"] = pd.to_numeric(prod_df["Production_Output"], errors="coerce").fillna(0.0)

        prod_agg = (prod_df

                    .groupby(["run", "seed", "Investor_ID"], as_index=False)

                    .agg(Production_Output=("Production_Output", "sum")))

 

        # Totals per (run, seed) for market share calculation

        totals = (prod_agg.groupby(["run", "seed"], as_index=False)["Production_Output"]

                .sum()

                .rename(columns={"Production_Output": "total_run_output"}))

 

        # Merge optimism with production (include investors even if they produced zero)

        df = (optim

            .merge(prod_agg, on=["run", "seed", "Investor_ID"], how="left")

            .merge(totals, on=["run", "seed"], how="left"))

 

        # Zero if missing production; compute market share

        df["Production_Output"] = df["Production_Output"].fillna(0.0)

        # Avoid division by zero (no output in a run)

        df["market_share"] = df["Production_Output"] / df["total_run_output"].replace({0: np.nan})

 

        # --- Min–max scale Optimism_Factor within each (run, seed) so overlays are comparable ---

        def _minmax(series: pd.Series) -> pd.Series:

            s = pd.to_numeric(series, errors="coerce")

            mn, mx = s.min(), s.max()

            if not np.isfinite(mn) or not np.isfinite(mx):

                return pd.Series(np.nan, index=s.index)

            if mx == mn:

                # All same optimism in this run → center at 0.5 so it still shows up

                return pd.Series(0.5, index=s.index)

            return (s - mn) / (mx - mn)

 

        df["opt_norm"] = df.groupby(["run", "seed"])["Optimism_Factor"].transform(_minmax)

 

        # Clean up rows that can't be plotted

        df = df.dropna(subset=["opt_norm", "market_share"])

 

        if df.empty:

            return None

 

        # A friendly legend key

        df["run_seed"] = df["run"].astype(str) + " | " + df["seed"].astype(str)

 

        # Build the figure

        fig = px.scatter(

            df,

            x="opt_norm",

            y="market_share",

            color="run_seed",   # overlay runs; each run|seed has its own color

            hover_name="Investor_ID",

            hover_data={

                "Optimism_Factor": ":.3f",

                "opt_norm": ":.2f",

                "market_share": ":.2%",

                "Production_Output": ":.0f",

                "run": True,

                "seed": True,

            },

            labels={

                "opt_norm": "Optimism (min–max within run)",

                "market_share": "Market Share (of run)",

                "run_seed": "Run | Seed",

            },

            opacity=0.95,

        )

 

        fig.update_layout(

            template="plotly_white",

            legend_title_text="Run | Seed",

            xaxis=dict(range=[-0.02, 1.02]),

            yaxis=dict(tickformat=".0%"),

            margin=dict(l=20, r=20, t=30, b=40),

        )

 

        # Optional: add a lowess trend to visualize overall correlation (requires statsmodels)

        # import plotly.express as px

        # trend_fig = px.scatter(df, x="opt_norm", y="market_share", trendline="lowess")

        # You could extract the trendline trace and add it: fig.add_traces(trend_fig.data[-1])

 

        return fig

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

    # --------------------Feedstock Metrics --------------------

 

    @app.callback(

        Output("graph-feedstock-used", "figure"),

        Input("store-batch-feedstock-aggregator", "data"),

    )

    def plot_feedstock_used(faggregator_data):

        if not faggregator_data:

            return None

 

        df = pd.DataFrame(faggregator_data)

 

 

        df["Feedstock_Used"] = df["Max_Supply"] - df["Available_Feedstock"]

 

        # Group by AgentID and calculate mean and std of feedstock used

        usage_stats = df.groupby("AgentID")["Feedstock_Used"].agg(["mean", "std"]).reset_index()

 

        # Get max supply per AgentID

        max_supply = df.groupby("AgentID")["Max_Supply"].max().reset_index()

 

        # Merge both

        merged = pd.merge(usage_stats, max_supply, on="AgentID")

 

        # Create figure

        fig = go.Figure()

 

        # Bar for average feedstock used

        fig.add_trace(go.Bar(

            x=merged["AgentID"],

            y=merged["mean"],

            error_y=dict(type='data', array=merged["std"]),

            name="Average Feedstock Used",

            marker_color="#0400ff"

        ))

 

        # Transparent bar for max supply

        fig.add_trace(go.Bar(

            x=merged["AgentID"],

            y=merged["Max_Supply"],

            name="Max Feedstock Available",

            marker_color="#3482b9", opacity=0.3

        ))

 

        fig.update_layout(

            margin=dict(t=20, b=40, l=40, r=20),

            xaxis_title="State Aggregator",

            yaxis_title="Feedstock Quantity",

            barmode='overlay'

        )

 

        return fig

 

 

 

 

# --------------------- Econometric Metrics ---------------------

 

 

    @app.callback(

        Output("graph-roace-kpi", "figure"),

        Input("store-batch-investor", "data"),

    )

    def kpi_roace_overall(model_data):

 

        # Empty state

        if not model_data:

            return None

 

        df = pd.DataFrame(model_data)

 

        # Per-run mean ROACE

        run_means = df.groupby("run", dropna=False)["Raw_ROACE"].mean().dropna()

 

        # Aggregate statistics

        mean_val = float(run_means.mean())

        p10 = float(run_means.quantile(0.10)) if len(run_means) > 0 else np.nan

        p90 = float(run_means.quantile(0.90)) if len(run_means) > 0 else np.nan

 

        subtitle = (

            f"P10–P90: {p10:.0%}–{p90:.0%}"

            if pd.notna(p10) and pd.notna(p90)

            else "P10–P90: n/a"

        )

 

        # Gauge axis: symmetric around 0, min span ±10%

        roace_vals = df["Raw_ROACE"].to_numpy(dtype=float)

        try:

            max_abs_val = float(np.nanmax(np.abs(roace_vals)))

        except ValueError:

            max_abs_val = 0.0

        max_abs_val = max(0.10, max_abs_val)  # at least ±10%

        axis_span = max_abs_val * 1.2         # 20% padding

        axis_min = -axis_span * 100.0         # percent units

        axis_max = axis_span * 100.0

 

        steps = []

        if axis_min < 0:

            steps.append({"range": [axis_min, min(0.0, axis_max)], "color": "rgba(220, 53, 69, 0.15)"})  # red-ish

        amber_upper = min(axis_max, 10.0)

        if axis_max > 0 and amber_upper > 0:

            steps.append({"range": [0.0, amber_upper], "color": "rgba(255, 193, 7, 0.15)"})              # amber

        if axis_max > 10.0:

            steps.append({"range": [10.0, axis_max], "color": "rgba(40, 167, 69, 0.15)"})                # green

 

        # KPI value (percent)

        value_pct = 100.0 * mean_val

 

        fig = go.Figure(go.Indicator(

            mode="gauge+number",

            value=value_pct,

            number={"suffix": "%", "font": {"size": 40}},

            gauge={

                "axis": {"range": [axis_min, axis_max]},

                "bar": {"color": "#1f77b4"},

                "bgcolor": "rgba(0,0,0,0)",

                "borderwidth": 0,

                "steps": steps,

                # Threshold: set a target (e.g., 10% ROACE). Adjust as you prefer.

                "threshold": {"line": {"color": "#28a745", "width": 2}, "thickness": 0.75, "value": 8},

            },

            title={

                "text": f"Overall ROACE&lt;br&gt;&lt;span style='font-size:0.8em;color:#666'&gt;{subtitle}&lt;/span&gt;"

            }

        ))

 

        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=60, b=10))

        return fig

 

    @app.callback(

        Output("graph-roace-by-year", "figure"),

        Input("store-batch-investor", "data"),

    )

    def graph_roace_by_year(investor_data):

        """

        Average ROACE by Year (across runs).

        Steps:

        1) Within each (run, Year), average ROACE across investors.

        2) Across runs, average those run-year means per Year (equal weighting).

        3) Show mean line and an optional P10–P90 shaded band across runs.

        """

 

        if not investor_data:

            return None

 

        df = pd.DataFrame(investor_data)

 

        # 1) Investor mean per (run, Year)

        #    If there are multiple rows per investor per year, we average them first.

        per_investor = (

            df.groupby(["run", "Year", "AgentID"], dropna=False)["Raw_ROACE"]

            .mean()

            .reset_index(name="ROACE_investor")

        )

 

        # 2) For each (run, Year), average across investors

        per_run_year = (

            per_investor.groupby(["run", "Year"], dropna=False)["ROACE_investor"]

            .mean()

            .reset_index(name="ROACE_run_mean")

        )

 

        # 3) Across runs per Year: mean and P10–P90

        def q10(x): return float(np.nanpercentile(x, 10)) if len(x) else np.nan

        def q90(x): return float(np.nanpercentile(x, 90)) if len(x) else np.nan

 

        by_year = (

            per_run_year.groupby("Year", dropna=False)["ROACE_run_mean"]

            .agg(mean="mean", p10=q10, p90=q90, runs="size")

            .reset_index()

            .sort_values("Year")

        )

 

        by_year["mean_pct"] = by_year["mean"] * 100.0

        by_year["p10_pct"] = by_year["p10"] * 100.0

        by_year["p90_pct"] = by_year["p90"] * 100.0

 

        # Y-axis range: symmetric around 0, min ±10%, with padding

        y_candidates = np.concatenate([

            by_year["mean"].to_numpy(dtype=float),

            by_year["p10"].to_numpy(dtype=float),

            by_year["p90"].to_numpy(dtype=float),

        ], axis=0)

        y_candidates = y_candidates[~np.isnan(y_candidates)]

        if y_candidates.size == 0:

            axis_min, axis_max = -10.0, 10.0

        else:

            max_abs = max(0.10, float(np.nanmax(np.abs(y_candidates))))

            span = max_abs * 1.2  # 20% padding

            axis_min, axis_max = -span * 100.0, span * 100.0

 

        # Build figure

        fig = go.Figure()

 

        # Optional P10–P90 band (only if meaningful)

        has_band = (

            (by_year["runs"].max() >= 2)

            and by_year["p10_pct"].notna().any()

            and by_year["p90_pct"].notna().any()

        )

        if has_band:

            fig.add_trace(go.Scatter(

                x=by_year["Year"],

                y=by_year["p90_pct"],

                mode="lines",

                line=dict(width=0),

                hoverinfo="skip",

                showlegend=False,

                name="P10–P90 band (upper)",

            ))

            fig.add_trace(go.Scatter(

                x=by_year["Year"],

                y=by_year["p10_pct"],

                mode="lines",

                line=dict(width=0),

                fill="tonexty",

                fillcolor="rgba(31, 119, 180, 0.10)",  # light blue band

                hoverinfo="skip",

                showlegend=False,

                name="P10–P90 band (lower)",

            ))

 

        # Mean line

        fig.add_trace(go.Scatter(

            x=by_year["Year"],

            y=by_year["mean_pct"],

            mode="lines+markers",

            line=dict(color="#1f77b4", width=3),

            marker=dict(size=6, color="#1f77b4"),

            name="Mean ROACE across runs",

            hovertemplate="<b>Year %{x}</b><br>Mean ROACE: %{y:.1f}%<extra></extra>",

        ))

 

        max_runs = int(by_year["runs"].max()) if "runs" in by_year.columns else None

 

        fig.update_layout(

            template="plotly_white",

            margin=dict(l=10, r=10, t=60, b=10),

            xaxis=dict(title="Year", tickmode="linear"),

            yaxis=dict(title="ROACE", range=[axis_min, axis_max], ticksuffix="%", zeroline=True, zerolinewidth=1, zerolinecolor="#ccc"),

            showlegend=False,

        )

 

        return fig

 

 

 

 

 

 

 

 

 

 

 

 

 

 

# ---------- Store graphs ----------

    # --- SAF Price Over Time ---

    @app.callback(

        Output("store-graph-saf-price-over-time", "data"),

        Input("graph-saf-price-over-time", "figure"),

    )

    def save_saf_price_figure(fig):

        if not fig:

            raise PreventUpdate

        return fig

 

    # --- Supply vs Demand ---

    @app.callback(

        Output("store-graph-supply-demand", "data"),

        Input("graph-supply-demand", "figure"),

    )

    def save_supply_demand_figure(fig):

        if not fig:

            raise PreventUpdate

        return fig

 

    # --- Consumer Price Forecast ---

    @app.callback(

        Output("store-consumer-price-forecast-graph", "data"),

        Input("consumer-price-forecast-graph", "figure"),

    )

    def save_consumer_price_forecast_figure(fig):

        if not fig:

            raise PreventUpdate

        return fig

 

    # --- Owned Sites per Investor ---

    @app.callback(

        Output("store-graph-owned-sites-per-investor", "data"),

        Input("graph-owned-sites-per-investor", "figure"),

    )

    def save_owned_sites_figure(fig):

        if not fig:

            raise PreventUpdate

        return fig

 

    # --- Plant EBIT ---

    @app.callback(

        Output("store-plant-ebit-graph", "data"),

        Input("plant-ebit-graph", "figure"),

    )

    def save_plant_ebit_figure(fig):

        if not fig:

            raise PreventUpdate

        return fig

 

    # --- Investor KPIs ---

    @app.callback(

        Output("store-investor-kpis-graph", "data"),

        Input("investor-kpis-graph", "figure"),

    )

    def save_investor_kpis_figure(fig):

        if not fig:

            raise PreventUpdate

        return fig

 

    # --- Supply by Plant ---

    @app.callback(

        Output("store-graph-supply-by-plant", "data"),

        Input("graph-supply-by-plant", "figure"),

    )

    def save_supply_by_plant_figure(fig):

        if not fig:

            raise PreventUpdate

        return fig

 

    # --- Feedstock Availability ---

    @app.callback(

        Output("store-feedstock-availability-graph", "data"),

        Input("feedstock-availability-graph", "figure"),

    )

    def save_feedstock_availability_figure(fig):

        if not fig:

            raise PreventUpdate

        return fig

 

 

 

 

 