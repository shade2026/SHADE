import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from imputation_models import DLinear, iTransformer
from utils.spacefeatures import space_features
from utils.timefeatures import time_features
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# 为了验证测试集原始掩码部分的补全效果，需重建数据集
class my_dataset(Dataset):
    def __init__(self, args, root_path, win_size, original_mask, mask, step=1, features='S', target='OT',
                 scale=True, inverse=False, timeenc=1, freq='h', cols=None):
        self.args = args
        self.step = step
        self.win_size = win_size  # seq_len
        # init
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = 'test'

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path

        self.original_mask = original_mask  # 原始掩码矩阵
        self.mask = mask  # 更新后的掩码矩阵
        self.__read_data__()

    def __read_data__(self):
        ### 以下代码只加载测试集 ###
        self.scaler = StandardScaler()
        data_name = self.args.data_path.split('_')[0]
        df_real = pd.read_csv(os.path.join(self.root_path, data_name + '.csv'))  # 正常数据集
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
        border1 = border1s[2]
        border2 = border2s[2]

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
        data_space_stamp = np.array(space_features(data_space_stamp, data_name))

        self.data_false = data_false[border1:border2]
        if self.inverse:
            self.data_true = df_true.values[border1:border2]
        else:
            self.data_true = data_true[border1:border2]
        self.data_time_stamp = data_time_stamp
        self.data_space_stamp = data_space_stamp

    def __getitem__(self, index):
        index = index * self.step
        begin = index
        end = index + self.win_size

        seq_x = self.data_false[begin:end][:, :, None]
        seq_y = self.data_true[begin:end][:, :, None]
        seq_x_mark = self.data_time_stamp[begin:end]
        space_mark_x = np.tile(self.data_space_stamp, (self.win_size, 1, 1))
        original_mask = self.original_mask[begin:end][:, :, None]
        mask = self.mask[begin:end][:, :, None]

        return seq_x, seq_y, seq_x_mark, space_mark_x, original_mask, mask

    def __len__(self):
        return (self.num_test - self.win_size) // self.step + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Exp_DL:
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'DLinear': DLinear,
            'iTransformer': iTransformer,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        print(f'Model Name: {self.args.imputation_model}')
        model = self.model_dict[self.args.imputation_model].Model(self.args).float()

        setting = '{}_{}_{}_{}_sl{}_dm{}_el{}_dl{}_df{}_ar{}_mr{}_ip{}_fc{}_eb{}_{}_{}'.format(
            "imputation",
            self.args.model_id,
            self.args.imputation_model,
            self.args.data,
            self.args.seq_len,
            self.args.d_model,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.anomaly_ratio,
            self.args.mask_rate,
            self.args.interpolate,
            self.args.factor,
            self.args.embed,
            self.args.des, 0)

        print('loading model')
        model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        model.load_state_dict(torch.load(model_path))  # 必须先准备训练好的模型

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, original_mask, mask, scale=True, inverse=False, timeenc=1, freq='h'):
        dataset = my_dataset(self.args, self.args.root_path, self.args.seq_len, original_mask, mask, step=1)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False
        )
        return dataloader

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=1e-3)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def test(self, original_mask, mask):
        # origin_mask: 原始掩码矩阵
        # mask: 修改后的掩码矩阵
        start_time = time.time()
        test_loader = self._get_data(original_mask, mask)

        preds = []
        trues = []
        original_masks = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, batch_x_time_mark, batch_x_space_mark, original_mask, mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_time_mark = batch_x_time_mark.float().to(self.device)
                batch_x_space_mark = batch_x_space_mark.float().to(self.device)
                original_mask = original_mask.to(self.device)
                mask = mask.to(self.device)

                # random mask
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, :, f_dim:]
                mask = mask[:, :, :, f_dim:]
                original_mask = original_mask[:, :, :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                original_masks.append(original_mask.detach().cpu())

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        original_masks = np.concatenate(original_masks, 0)

        # 测试时根据output和data_true的原始掩码部分进行评估
        mae, mse, rmse, mape, mspe = metric(preds[original_masks == 0], trues[original_masks == 0])
        print("Total cost time: {}".format(time.time() - start_time))
        print("MAE: {0:.7f} MSE: {1:.7f} RMSE: {2:.7f} MAPE: {3:.7f} MSPE: {4:.7f}".format(mae, mse, rmse, mape, mspe))

        f = open("result_anomaly_detection.txt", 'a')
        f.write("Deep Learning Imputation Testing  \n")
        f.write(
            "MAE: {0:.7f} MSE: {1:.7f} RMSE: {2:.7f} MAPE: {3:.7f} MSPE: {4:.7f}".format(mae, mse, rmse, mape, mspe))
        f.write('\n')
        f.close()
