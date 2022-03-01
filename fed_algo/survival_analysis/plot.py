import numpy as np
import plotly.graph_objects as go
from lifelines import utils


def plot_km_plotly(data, category=False):
    fig = go.Figure()
    kms = [x for x in data.columns]
    for km in kms:
        if len(kms) >= 1:
            timeline = data[km].dropna().reset_index()["timeline"]
            values = data[km].dropna().reset_index()[km]
            if category:
                fig.add_trace(go.Scatter(name=f'{category}={km}', x=timeline, y=values, mode='lines', line_shape='hv'))
            else:
                fig.add_trace(go.Scatter(name=km, x=timeline, y=values, mode='lines', line_shape='hv'))

    fig.layout = {
        "xaxis": {
            "title": "timeline",
        },
        "yaxis": {
            "title": "Survival Probability S(x)"
        },
        "template": "simple_white",
    }
    fig.update_layout(showlegend=True)

    return fig.to_json()


def plot_na_plotly(data, category=False):
    fig = go.Figure()
    nas = [x for x in data.columns]
    for na in nas:
        if len(nas) >= 1:
            timeline = data[na].dropna().reset_index()["timeline"]
            values = data[na].dropna().reset_index()[na]
            if category:
                fig.add_trace(go.Scatter(name=f'{category}={na}', x=timeline, y=values, mode='lines', line_shape='hv'))
            else:
                fig.add_trace(go.Scatter(name=na, x=timeline, y=values, mode='lines', line_shape='hv'))
    fig.layout = {
        "xaxis": {
            "title": "timeline",
        },
        "yaxis": {
            "title": "Hazard Ratio h(x)"
        },
        "template": "simple_white"

    }
    fig.update_layout(showlegend=True)

    return fig.to_json()


def plot_cox_plotly(params_, standard_errors_, alpha, columns=None):
    z = utils.inv_normal_cdf(1 - alpha / 2)
    user_supplied_columns = True

    if columns is None:
        user_supplied_columns = False
        columns = params_.index

    log_hazards = params_.loc[columns].values.copy()
    order = list(range(len(columns) - 1, -1, -1)) if user_supplied_columns else np.argsort(log_hazards)
    symmetric_errors = z * standard_errors_[columns].values

    fig = go.Figure()
    # Use x instead of y argument for horizontal plot
    for i in range(len(symmetric_errors)):
        log_hazard = log_hazards[order][i]
        symmetric_error = symmetric_errors[order][i]
        fig.add_trace(
            go.Box(name=columns[order][i], x=[log_hazard - symmetric_error, log_hazard, log_hazard + symmetric_error]))
    fig.layout = {
        "autosize": True,
        "template": "simple_white",
        "xaxis": {
            "title": "log(HR) (95% CI)",
        },
    }
    fig.add_vline(x=0, opacity=1, line_width=0.5, line_dash="dash")

    return fig.to_json()
