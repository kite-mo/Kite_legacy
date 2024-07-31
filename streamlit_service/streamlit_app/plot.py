import plotly.express as px
import plotly.graph_objects as go

import pandas as pd


def get_timelength_distribution_fig(wafer_list):
    wafer_length_list = [len(wafer) for wafer in wafer_list]
    wafer_length_df = pd.DataFrame(wafer_length_list, columns=["time_lengths"])

    fig = px.histogram(data_frame=wafer_length_df, x="time_lengths")
    return fig
