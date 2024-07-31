import pandas as pd
import numpy as np

from typing import Any, Dict, List, Tuple, Union


def read_time_series_df(upload_files, time_col):
    df = pd.read_csv(upload_files)
    df[time_col] = pd.to_datetime(df[time_col], format="mixed")
    df = df.sort_values(by=time_col).reset_index(drop=True)
    return df


def get_wafer_list_sorted_by_time(df, key_list, time_col):
    infos = [info for info, wafer in df.groupby(key_list)]
    wafers = [wafer for info, wafer in df.groupby(key_list)]

    first_time_list = []

    for wafer in wafers:
        wafer["time_col"] = pd.to_datetime(wafer[time_col])
        wafer = wafer.reset_index(drop=True)
        first_time_list.append(wafer["time_col"].iloc[0])

    sorted_index_list = np.argsort(first_time_list)

    sorted_wafer_list = list(map(lambda i: wafers[i], sorted_index_list))
    sorted_info_list = list(map(lambda i: infos[i], sorted_index_list))
    sorted_first_time_list = list(map(lambda i: first_time_list[i], sorted_index_list))

    return sorted_wafer_list, sorted_info_list, sorted_first_time_list


def get_info_df(df, data_unit_list, meta_column_list):
    # essential basic information
    time_col = [col for col in meta_column_list if "time" == col][0]
    unit_counts = len(data_unit_list)
    start_time, end_time = (
        df.loc[0, time_col],
        df.loc[len(df) - 1, time_col],
    )
    time_measurement = np.round(
        np.mean(
            [
                np.mean(np.diff(data_unit[time_col]) / np.timedelta64(1, "s"))
                for data_unit in data_unit_list
            ]
        ),
        4,
    )
    info_dict = {
        "Data Unit Counts": unit_counts,
        "Start Time": start_time,
        "End Time": end_time,
        "Time Measurements (sec)": time_measurement,
    }

    # other info columns
    other_info_cols = [col for col in meta_column_list if "time" != col]
    selected_info_cols = []
    for col in other_info_cols:
        if len(df[col].unique()) == 1:
            selected_info_cols.append(col)

    for col in selected_info_cols:
        info_dict[col] = df[col].unique()[0]

    info_df = pd.DataFrame(
        info_dict,
        index=["Value"],
    ).T

    return info_df


def convert_result_dict_to_scoring_df(result_dict):
    df_row_list = []
    for key, result in result_dict.items():
        first_time, wafer_key = key
        prediction_result, scoring_result = result["prediction"], result["scoring"]

        converted_df = pd.DataFrame(
            {
                "time": first_time,
                "wafer_key": str(wafer_key),
                "wafer_score": scoring_result["waferScore"],
            },
            index=[0],
        )

        for sensor, sensor_score in scoring_result["sensorScore"].items():
            converted_df[f"{sensor}"] = sensor_score
        df_row_list.append(converted_df)

    concat_score_df = pd.concat(df_row_list, axis=0).reset_index(drop=True)
    concat_score_df["label"] = "good"
    concat_score_df.loc[concat_score_df["wafer_score"] < 80, "label"] = "bad"
    return concat_score_df


def get_result_key_list_from_scoring_df(scoring_df: pd.DataFrame):
    result_key_list = []
    for i, row in scoring_df.iterrows():
        time, key, wafer_score = (
            row["time"],
            row["wafer_key"],
            round(row["wafer_score"], 3),
        )
        new_key = tuple([time, eval(key), wafer_score])
        result_key_list.append(new_key)
    return result_key_list


def get_sensor_key_list_from_scoring_df(
    scoring_df: pd.DataFrame, result_key: tuple, hm_column_list: List
):
    sensor_key_list = []

    time, key = result_key
    selected_score_row = scoring_df[
        (scoring_df["time"] == pd.to_datetime(time))
        & (scoring_df["wafer_key"] == str(key))
    ].reset_index(drop=True)
    selected_score_df = selected_score_row[hm_column_list].T.sort_values(0)

    for sensor, score in selected_score_df.iterrows():
        score = score.values[0]
        sensor_key_list.append(tuple([sensor, score]))

    return sensor_key_list
