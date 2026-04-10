import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.spacefeatures import space_features
# from data_provider.m4 import M4Dataset, M4Meta
# from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_Anomalous_Humidity(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag='train', features='S', target='OT', scale=True,
                 inverse=False, timeenc=1, freq='h', cols=None):
        self.args = args
        self.flag = flag
        self.step = step
        self.win_size = win_size  # seq_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_real = pd.read_csv(os.path.join(self.root_path, 'Humidity.csv'))  # 正常数据集
        df_raw = pd.read_csv(os.path.join(self.root_path, self.args.data_path + '.csv'))  # 异常数据集
        cols = list(df_raw.columns)
        cols.remove('time')
        cols.remove('week')
        df_raw = df_raw[['time'] + ['week'] + cols]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        border1s = [0, self.num_train, len(df_raw) - self.num_test]
        border2s = [self.num_train, self.num_train + self.num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[2:]
        df_true = df_real[cols_data]
        df_false = df_raw[cols_data]

        if self.scale:
            # 正常数据集
            train_data = df_true[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_true = self.scaler.transform(df_true.values)
            # 异常数据集
            train_data = df_false[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_false = self.scaler.transform(df_false.values)
        else:
            data_true = df_true.values
            data_false = df_false.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        data_time_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        data_time_stamp = data_time_stamp.transpose(1, 0)

        location = df_raw.columns[2:].values
        data_space_stamp = []
        for loc in location:
            location_stamp = []
            for loc_string in loc.split(','):
                loc_int = float(loc_string)
                location_stamp.append(loc_int)
            data_space_stamp.append(location_stamp)
        data_space_stamp = np.array(space_features(data_space_stamp, "Humidity"))

        self.data_false = data_false[border1:border2]
        if self.inverse:
            self.data_true = df_true.values[border1:border2]
        else:
            self.data_true = data_true[border1:border2]
        self.data_time_stamp = data_time_stamp
        self.data_space_stamp = data_space_stamp

        # mask
        self.mask_rate = self.args.mask_rate
        self.mask = np.random.rand(*(self.data_false.shape))
        self.mask[self.mask <= self.mask_rate] = 0  # masked
        self.mask[self.mask > self.mask_rate] = 1  # remained

    def __getitem__(self, index):
        index = index * self.step
        begin = index
        end = index + self.win_size

        seq_x = self.data_false[begin:end][:, :, None]
        seq_y = self.data_true[begin:end][:, :, None]
        seq_x_mark = self.data_time_stamp[begin:end]
        space_mark_x = np.tile(self.data_space_stamp, (self.win_size, 1, 1))
        mask = self.mask[begin:end][:, :, None]

        return seq_x, seq_y, seq_x_mark, space_mark_x, mask

    def __len__(self):
        if self.flag == "train":
            return (self.num_train - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.num_vali - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.num_test - self.win_size) // self.step + 1
        else:
            return (self.num_test - self.win_size) // self.win_size + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Anomalous_TrafficFlow(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag='train', features='S', target='OT', scale=True,
                 inverse=False, timeenc=1, freq='h', cols=None):
        self.args = args
        self.flag = flag
        self.step = step
        self.win_size = win_size  # seq_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_real = pd.read_csv(os.path.join(self.root_path, 'TrafficFlow.csv'))  # 正常数据集
        df_raw = pd.read_csv(os.path.join(self.root_path, self.args.data_path + '.csv'))  # 异常数据集
        cols = list(df_raw.columns)
        cols.remove('time')
        cols.remove('week')
        df_raw = df_raw[['time'] + ['week'] + cols]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        border1s = [0, self.num_train, len(df_raw) - self.num_test]
        border2s = [self.num_train, self.num_train + self.num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[2:]
        df_true = df_real[cols_data]
        df_false = df_raw[cols_data]

        if self.scale:
            # 正常数据集
            train_data = df_true[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_true = self.scaler.transform(df_true.values)
            # 异常数据集
            train_data = df_false[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_false = self.scaler.transform(df_false.values)
        else:
            data_true = df_true.values
            data_false = df_false.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        data_time_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        data_time_stamp = data_time_stamp.transpose(1, 0)

        location = df_raw.columns[2:].values
        data_space_stamp = []
        for loc in location:
            location_stamp = []
            for loc_string in loc.split(','):
                loc_int = float(loc_string)
                location_stamp.append(loc_int)
            data_space_stamp.append(location_stamp)
        data_space_stamp = np.array(space_features(data_space_stamp, "TrafficFlow"))

        self.data_false = data_false[border1:border2]
        if self.inverse:
            self.data_true = df_true.values[border1:border2]
        else:
            self.data_true = data_true[border1:border2]
        self.data_time_stamp = data_time_stamp
        self.data_space_stamp = data_space_stamp

        # mask
        self.mask_rate = self.args.mask_rate
        self.mask = np.random.rand(*(self.data_false.shape))
        self.mask[self.mask <= self.mask_rate] = 0  # masked
        self.mask[self.mask > self.mask_rate] = 1  # remained

    def __getitem__(self, index):
        index = index * self.step
        begin = index
        end = index + self.win_size

        seq_x = self.data_false[begin:end][:, :, None]
        seq_y = self.data_true[begin:end][:, :, None]
        seq_x_mark = self.data_time_stamp[begin:end]
        space_mark_x = np.tile(self.data_space_stamp, (self.win_size, 1, 1))
        mask = self.mask[begin:end][:, :, None]

        return seq_x, seq_y, seq_x_mark, space_mark_x, mask

    def __len__(self):
        if self.flag == "train":
            return (self.num_train - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.num_vali - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.num_test - self.win_size) // self.step + 1
        else:
            return (self.num_test - self.win_size) // self.win_size + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Anomalous_NO2(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag='train', features='S', target='OT', scale=True,
                 inverse=False, timeenc=1, freq='h', cols=None):
        self.args = args
        self.flag = flag
        self.step = step
        self.win_size = win_size  # seq_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_real = pd.read_csv(os.path.join(self.root_path, 'NO2.csv'))  # 正常数据集
        df_raw = pd.read_csv(os.path.join(self.root_path, self.args.data_path + '.csv'))  # 异常数据集
        cols = list(df_raw.columns)
        cols.remove('time')
        cols.remove('week')
        df_raw = df_raw[['time'] + ['week'] + cols]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        border1s = [0, self.num_train, len(df_raw) - self.num_test]
        border2s = [self.num_train, self.num_train + self.num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[2:]
        df_true = df_real[cols_data]
        df_false = df_raw[cols_data]

        if self.scale:
            train_data = df_true[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_true = self.scaler.transform(df_true.values)

            train_data = df_false[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_false = self.scaler.transform(df_false.values)
        else:
            data_true = df_true.values
            data_false = df_false.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        data_time_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        data_time_stamp = data_time_stamp.transpose(1, 0)

        location = df_raw.columns[2:].values
        data_space_stamp = []
        for loc in location:
            location_stamp = []
            for loc_string in loc.split(','):
                loc_int = float(loc_string)
                location_stamp.append(loc_int)
            data_space_stamp.append(location_stamp)
        data_space_stamp = np.array(space_features(data_space_stamp, "NO2"))

        self.data_false = data_false[border1:border2]
        if self.inverse:
            self.data_true = df_true.values[border1:border2]
        else:
            self.data_true = data_true[border1:border2]
        self.data_time_stamp = data_time_stamp
        self.data_space_stamp = data_space_stamp

        # mask
        self.mask_rate = self.args.mask_rate
        self.mask = np.random.rand(*(self.data_false.shape))
        self.mask[self.mask <= self.mask_rate] = 0  # masked
        self.mask[self.mask > self.mask_rate] = 1  # remained

    def __getitem__(self, index):
        index = index * self.step
        begin = index
        end = index + self.win_size

        seq_x = self.data_false[begin:end][:, :, None]
        seq_y = self.data_true[begin:end][:, :, None]
        seq_x_mark = self.data_time_stamp[begin:end]
        space_mark_x = np.tile(self.data_space_stamp, (self.win_size, 1, 1))
        mask = self.mask[begin:end][:, :, None]

        return seq_x, seq_y, seq_x_mark, space_mark_x, mask

    def __len__(self):
        if self.flag == "train":
            return (self.num_train - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.num_vali - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.num_test - self.win_size) // self.step + 1
        else:
            return (self.num_test - self.win_size) // self.win_size + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Anomalous_Temperature(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag='train', features='S', target='OT', scale=True,
                 inverse=False, timeenc=1, freq='h', cols=None):
        self.args = args
        self.flag = flag
        self.step = step
        self.win_size = win_size  # seq_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_real = pd.read_csv(os.path.join(self.root_path, 'Temperature.csv'))  # 正常数据集
        df_raw = pd.read_csv(os.path.join(self.root_path, self.args.data_path + '.csv'))  # 异常数据集
        cols = list(df_raw.columns)
        cols.remove('time')
        cols.remove('week')
        df_raw = df_raw[['time'] + ['week'] + cols]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        border1s = [0, self.num_train, len(df_raw) - self.num_test]
        border2s = [self.num_train, self.num_train + self.num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[2:]
        df_true = df_real[cols_data]
        df_false = df_raw[cols_data]

        if self.scale:
            # 正常数据集
            train_data = df_true[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_true = self.scaler.transform(df_true.values)
            # 异常数据集
            train_data = df_false[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_false = self.scaler.transform(df_false.values)
        else:
            data_true = df_true.values
            data_false = df_false.values

        df_stamp = df_raw[['time']][border1:border2]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)
        data_time_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        data_time_stamp = data_time_stamp.transpose(1, 0)

        location = df_raw.columns[2:].values
        data_space_stamp = []
        for loc in location:
            location_stamp = []
            for loc_string in loc.split(','):
                loc_int = float(loc_string)
                location_stamp.append(loc_int)
            data_space_stamp.append(location_stamp)
        data_space_stamp = np.array(space_features(data_space_stamp, "Temperature"))

        self.data_false = data_false[border1:border2]
        if self.inverse:
            self.data_true = df_true.values[border1:border2]
        else:
            self.data_true = data_true[border1:border2]
        self.data_time_stamp = data_time_stamp
        self.data_space_stamp = data_space_stamp

        # mask
        self.mask_rate = self.args.mask_rate
        self.mask = np.random.rand(*(self.data_false.shape))
        self.mask[self.mask <= self.mask_rate] = 0  # masked
        self.mask[self.mask > self.mask_rate] = 1  # remained

    def __getitem__(self, index):
        index = index * self.step
        begin = index
        end = index + self.win_size

        seq_x = self.data_false[begin:end][:, :, None]
        seq_y = self.data_true[begin:end][:, :, None]
        seq_x_mark = self.data_time_stamp[begin:end]
        space_mark_x = np.tile(self.data_space_stamp, (self.win_size, 1, 1))
        mask = self.mask[begin:end][:, :, None]

        return seq_x, seq_y, seq_x_mark, space_mark_x, mask

    def __len__(self):
        if self.flag == "train":
            return (self.num_train - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.num_vali - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.num_test - self.win_size) // self.step + 1
        else:
            return (self.num_test - self.win_size) // self.win_size + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
