import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np


def plot_compare_ref_by_sensor(select_ref_dict, sensor):
    color_list = px.colors.sequential.dense
    color_list = color_list[::-1][4:]

    # plot reference
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    upper_sigma = select_ref_dict["repMaxDict"][sensor]
    lower_sigma = select_ref_dict["repMinDict"][sensor]

    fig.add_trace(
        go.Line(
            x=list(range(0, len(upper_sigma))),
            y=upper_sigma,
            name=f"+ 3 Sigma",
            fill=None,
            line_color="gray",
        ),  # gray
        secondary_y=False,
    )
    fig.add_trace(
        go.Line(
            x=list(range(0, len(lower_sigma))),
            y=lower_sigma,
            name=f"-3 Sigma",
            fill="tonexty",
            line_color="gray",
        ),
        secondary_y=False,
    )

    ref_center = pd.Series(select_ref_dict["centerDict"][sensor])
    fig.add_trace(
        go.Line(
            x=ref_center.index,
            y=ref_center.values,
            name="AVG_Reference",
            line_color="red",
        ),
        secondary_y=False,
    )

    fig.update_layout(title_text=f"Sensor : {sensor}")
    fig.update_xaxes(title_text="indexes")
    fig.update_yaxes(title_text="values")
    fig.update_layout(width=1000, height=600)

    return fig


def get_time_scatter_fig(score_df, y_column):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=score_df["time"],
            y=score_df[y_column],
            mode="markers",
            marker=dict(
                color=np.where(score_df[y_column] >= 80, "blue", "red"),
            ),
        )
    )
    fig.update_yaxes(range=[0.0, 100])
    fig.add_hline(y=80.0, line_width=1.0, line_dash="dash", line_color="green")
    fig.update_xaxes(rangeslider_visible=True)

    if "score" not in y_column:
        title_text = y_column + " score"
    else:
        title_text = y_column

    fig.update_layout(title_text=title_text)
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Score")

    return fig


def plot_ref_target_score_ratio(reference_dict, inference_dict, score_dict, sensor):
    color_list = px.colors.sequential.dense
    color_list = color_list[::-1][4:]

    center_values = reference_dict["centerDict"][sensor]
    upper_sigma = reference_dict["repMaxDict"][sensor]
    lower_sigma = reference_dict["repMinDict"][sensor]

    target_values = inference_dict[sensor]["target"]
    point_score = score_dict["percentScore"][sensor]
    sensor_score = round(np.mean(point_score), 4)

    # plot reference
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Line(
            x=list(range(0, len(upper_sigma))),
            y=upper_sigma,
            name=f"+ 3 Sigma",
            fill=None,
            line_color="gray",
        ),  # gray
        secondary_y=False,
    )
    fig.add_trace(
        go.Line(
            x=list(range(0, len(lower_sigma))),
            y=lower_sigma,
            name=f"-3 Sigma",
            fill="tonexty",
            line_color="gray",
        ),
        secondary_y=False,
    )
    ref_center = pd.Series(center_values)

    fig.add_trace(
        go.Line(
            x=ref_center.index,
            y=ref_center.values,
            name="AVG_Reference",
            line_color="red",
        ),
        secondary_y=False,
    )

    target_values = pd.Series(target_values)
    fig.add_trace(
        go.Line(
            x=target_values.index,
            y=target_values.values,
            name="Target",
            line_color="blue",
        ),
        secondary_y=False,
    )

    score_ratio = pd.Series(point_score)
    fig.add_trace(
        go.Line(
            x=score_ratio.index,
            y=score_ratio.values,
            name="Score_Ratio",
            line_color="orange",
        ),
        secondary_y=True,
    )

    fig.add_hline(
        y=80.0, line_width=1.0, line_dash="dash", line_color="green", secondary_y=True
    )
    fig.update_layout(title_text=f"Sensor : {sensor}, Sensor Score : {sensor_score}")
    fig.update_yaxes(range=[0.0, 100], secondary_y=True)
    fig.update_xaxes(title_text="indexes")
    fig.update_yaxes(title_text="values")
    fig.update_yaxes(title_text="scores", secondary_y=True)

    return fig
