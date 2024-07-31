import sys, os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def sort_wafer_by_time(df: pd.DataFrame, key_list: list) -> list:

    wafers = np.array(list(df.groupby(key_list)))

    arg_list = []

    for info, temp in wafers:
        temp['time'] = pd.to_datetime(temp.time)
        temp = temp.reset_index(drop=True)
        arg_list.append(temp['time'].iloc[0])

    index_list = np.argsort(arg_list)
    sort_wafer = [(info, df.reset_index(drop=True)) for info, df in wafers[index_list]]

    return sort_wafer


def get_cycle_dict(wafer_df, wafer_columns, key_list, info_col):

    cycle_list = wafer_df.cycle.unique().tolist()

    cycle_dict = {}

    cycle_mean = {}
    cycle_std = {}

    for cycle in cycle_list:
        df = wafer_df[wafer_df.cycle == cycle]
        df_ = df.loc[:, wafer_columns]
        wafer_unit = sort_wafer_by_time(df_, key_list)
        # wafer_unit_ = list(
        #     filter(lambda x: norm_range[1] > len(x[1]) > norm_range[0], wafer_unit)
        # )

        w_mean = np.array(
            [
                wafer.iloc[:, (len(info_col) + len(key_list)) :].mean().values
                for info, wafer in wafer_unit
            ]
        )
        w_std = np.array(
            [
                wafer.iloc[:, (len(info_col) + len(key_list)) :].std().values
                for info, wafer in wafer_unit
            ]
        )

        c_mean = w_mean[0]
        cycle_mean[cycle] = c_mean

        c_std = w_std[0]
        cycle_std[cycle] = c_std

    cycle_dict['mean'] = cycle_mean
    cycle_dict['std'] = cycle_std

    return cycle_dict


def select_df(total_df, index_dict, type_='train'):  # 'valid', 'test'
    type_dict = index_dict[f'{type_}_idx']

    df_list = list()
    for cycle in type_dict:
        df_list.append(
            total_df[
                (total_df.cycle == cycle)
                & (total_df.id.isin(index_dict[f'{type_}_idx'][cycle]))
            ]
        )

    final_df = pd.concat(df_list)
    final_df = final_df.reset_index(drop=True)
    return final_df
