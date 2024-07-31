import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class dataset_base_ae(Dataset):
    def __init__(self, wafer_unit, max_len, id_dict, f_list):

        super(dataset_base_ae, self).__init__()
        self.wafer_unit = wafer_unit
        self.max_len = max_len
        self.id_dict = id_dict
        self.f_list = f_list

    def __getitem__(self, idx):

        info_t = self.wafer_unit[idx][0]
        wafer_t = self.wafer_unit[idx][1]
        id_t = wafer_t.cycle.unique()[0]

        c_mean_t = self.id_dict['mean'][id_t]
        c_std_t = self.id_dict['std'][id_t]

        torr_t = wafer_t.torr.unique()[0]
        y = wafer_t.leak.unique()[0]

        wafer_ = wafer_t.loc[:, self.f_list].reset_index(drop=True)
        c_mean, c_std = c_mean_t, c_std_t

        wafer_df = np.array(wafer_)
        x = pd.DataFrame((wafer_df - c_mean) / (c_std))
        last_row = x.iloc[-1, :].values

        if len(x) < self.max_len:
            add_num = abs(len(x) - self.max_len)
            for num in range(add_num):
                x.loc[len(x)] = last_row

        X = np.array(x).T

        return X, y, torr_t, id_t

    def __len__(self):
        return len(self.wafer_unit)
