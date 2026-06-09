"""Plotly figures for the LabelLift Streamlit dashboard.

These helpers turn aggregated dataframes into investor-readable charts. They never
load data themselves — the dashboard passes in prepared frames so every figure is
fast to render even on the full synthetic dataset.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# A small, consistent palette so the dashboard reads as one product.
COLOR_OBSERVED = "#9aa7b8"  # muted slate — the censored view
COLOR_LABELLIFT = "#2f6df6"  # LabelLift blue — the corrected view
COLOR_TRUE = "#13b981"  # green — synthetic ground truth
COLOR_FUNNEL = "#2f6df6"


def label_problem_funnel(stage_counts: dict[str, int]) -> go.Figure:
    """Funnel of how many transactions survive each censorship stage.

    Args:
        stage_counts: Ordered mapping of stage label -> count.
    """
    labels = list(stage_counts.keys())
    values = list(stage_counts.values())
    fig = go.Figure(
        go.Funnel(
            y=labels,
            x=values,
            textinfo="value+percent initial",
            marker={"color": COLOR_FUNNEL},
            connector={"line": {"color": COLOR_OBSERVED}},
        )
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=420,
        title="How many transactions survive to become a fraud label",
    )
    return fig


def blindspot_scatter(atlas: pd.DataFrame) -> go.Figure:
    """Scatter of observed vs. corrected fraud rate per issuer.

    Points above the diagonal are issuers whose true fraud is most undercounted by
    the raw observed labels. Marker size encodes transaction volume.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=atlas["observed_fraud_rate"],
            y=atlas["corrected_fraud_rate"],
            mode="markers",
            marker={
                "size": atlas["transaction_count"],
                "sizemode": "area",
                "sizeref": 2.0 * atlas["transaction_count"].max() / (40.0**2),
                "sizemin": 4,
                "color": atlas["blindspot_multiplier"],
                "colorscale": "Blues",
                "showscale": True,
                "colorbar": {"title": "Blindspot<br>multiplier"},
                "line": {"width": 1, "color": COLOR_LABELLIFT},
            },
            text=atlas["issuer_id"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "observed fraud rate: %{x:.3%}<br>"
                "corrected fraud rate: %{y:.3%}<extra></extra>"
            ),
        )
    )
    upper = float(
        max(atlas["observed_fraud_rate"].max(), atlas["corrected_fraud_rate"].max())
    ) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, upper],
            y=[0, upper],
            mode="lines",
            line={"dash": "dash", "color": COLOR_OBSERVED},
            name="observed = corrected",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=460,
        xaxis={"title": "Observed fraud rate (what the model sees)", "tickformat": ".2%"},
        yaxis={"title": "LabelLift corrected fraud rate", "tickformat": ".2%"},
        title="Per-issuer fraud blind spots",
        showlegend=False,
    )
    return fig


def pseudo_label_histogram(scores: pd.DataFrame) -> go.Figure:
    """Overlaid histogram of baseline scores vs. LabelLift pseudo-labels."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scores["baseline_fraud_score"],
            name="Baseline fraud score",
            opacity=0.65,
            marker_color=COLOR_OBSERVED,
            nbinsx=60,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=scores["labellift_pseudo_label"],
            name="LabelLift pseudo-label",
            opacity=0.65,
            marker_color=COLOR_LABELLIFT,
            nbinsx=60,
        )
    )
    fig.update_layout(
        barmode="overlay",
        margin={"l": 10, "r": 10, "t": 50, "b": 70},
        height=420,
        yaxis={"title": "Transactions", "type": "log"},
        xaxis={"title": "Score"},
        title="Baseline vs. LabelLift score distribution (log scale)",
        legend={"orientation": "h", "y": -0.25, "x": 0},
    )
    return fig


def backtest_bar(metrics: pd.DataFrame) -> go.Figure:
    """Grouped bar comparing the two models on the headline ranking metrics."""
    display_metrics = [
        ("average_precision", "Average precision"),
        ("precision_at_1_percent", "Precision @ 1%"),
        ("recall_at_1_percent", "Recall @ 1%"),
    ]
    indexed = metrics.set_index("model")
    model_styles = {
        "raw_observed_label_model": ("Raw observed labels", COLOR_OBSERVED),
        "labellift_pseudo_label_model": ("LabelLift pseudo-labels", COLOR_LABELLIFT),
    }
    fig = go.Figure()
    x_labels = [label for _, label in display_metrics]
    for model_key, (model_label, color) in model_styles.items():
        if model_key not in indexed.index:
            continue
        row = indexed.loc[model_key]
        fig.add_trace(
            go.Bar(
                name=model_label,
                x=x_labels,
                y=[float(row[col]) for col, _ in display_metrics],
                marker_color=color,
                text=[f"{float(row[col]):.3f}" for col, _ in display_metrics],
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        margin={"l": 10, "r": 10, "t": 50, "b": 60},
        height=440,
        yaxis={"title": "Score (higher is better)"},
        title="Recovering true fraud: raw labels vs. LabelLift",
        legend={"orientation": "h", "y": -0.15, "x": 0},
    )
    return fig
