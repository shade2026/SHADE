import pandas as pd
from sklearn.preprocessing import StandardScaler
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


class Exp_DMF:
    def __init__(self, args):
        self.args = args
        # DMF configuration
        self.dim_z = 16  # dim_z
        self.hidden_layer = 20
        self.hidden_layer1 = 24
        self.output_dim = 32
        self.beta = 0.001
        self.lmda = 0.001

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.dim_z, self.hidden_layer),
            nn.Tanh(),
            nn.Linear(self.hidden_layer, self.hidden_layer1),
            nn.Tanh(),
            nn.Linear(self.hidden_layer1, self.output_dim)
        )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, mask, scale=True, inverse=False, timeenc=1, freq='h'):
        self.scaler = StandardScaler()
        data_name = self.args.data_path.split('_')[0]
        df_real = pd.read_csv(os.path.join(self.args.root_path, data_name + '.csv'))  # 正常数据集
        df_raw = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path + '.csv'))  # 异常数据集

        cols = list(df_raw.columns)
        cols.remove('time')
        cols.remove('week')
        df_raw = df_raw[['time'] + ['week'] + cols]

        self.num_train = int(len(df_raw) * 0.7)
        self.num_test = int(len(df_raw) * 0.2)
        self.num_vali = len(df_raw) - self.num_train - self.num_test
        border1s = [0, self.num_train, len(df_raw) - self.num_test]
        border2s = [self.num_train, self.num_train + self.num_vali, len(df_raw)]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[2:]
        df_true = df_real[cols_data]
        df_false = df_raw[cols_data]

        if scale:
            train_data = df_true[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_false = self.scaler.transform(df_false.values)
            data_true = self.scaler.transform(df_true.values)
        else:
            data_true = df_true.values
            data_false = df_false.values

        self.data_false = data_false[border1:border2]
        if inverse:
            self.data_true = df_true.values[border1:border2]
        else:
            self.data_true = data_true[border1:border2]

        # mask
        self.mask = mask

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

    def test(self, original_mask, mask, train_epochs=10):
        # origin_mask: 原始掩码矩阵
        # mask: 修改后的掩码矩阵
        start_time = time.time()
        self._get_data('test', mask)

        data_true = torch.Tensor(self.data_true).to(self.device)
        data_false = torch.Tensor(self.data_false).to(self.device)
        mask = torch.Tensor(mask).to(self.device)

        Z = nn.Parameter(torch.zeros(data_false.shape[0], self.dim_z, device=self.device))

        model_optim = self._select_optimizer()
        model_optim.add_param_group({'params': Z, 'lr': 1e-3})
        criterion = self._select_criterion()

        self.model.train()
        epoch_time = time.time()
        for epoch in range(train_epochs):
            model_optim.zero_grad()
            outputs = self.model(Z)
            regularization_loss = 0.0
            for param in self.model.parameters():
                regularization_loss += torch.sum(param ** 2)

            loss = criterion(outputs[mask == 1], data_false[mask == 1]) + self.lmda * regularization_loss
            loss.backward()
            model_optim.step()
            if (epoch + 1) % 1000 == 0:
                print("\tepoch: {0}, cost time: {1} | loss: {2:.7f}".format(epoch + 1, time.time() - epoch_time, loss))
                epoch_time = time.time()

        # 测试
        pred = self.model(Z)
        pred = pred.detach().cpu().numpy()
        true = data_true.detach().cpu().numpy()

        # 测试时根据output和data_true的原始掩码部分进行评估
        mae, mse, rmse, mape, mspe = metric(pred[original_mask == 0], true[original_mask == 0])
        print("Total cost time: {}".format(time.time() - start_time))
        print("MAE: {0:.7f} MSE: {1:.7f} RMSE: {2:.7f} MAPE: {3:.7f} MSPE: {4:.7f}".format(mae, mse, rmse, mape, mspe))

        f = open("result_anomaly_detection.txt", 'a')
        f.write("DMF Imputation Testing  \n")
        f.write(
            "MAE: {0:.7f} MSE: {1:.7f} RMSE: {2:.7f} MAPE: {3:.7f} MSPE: {4:.7f}".format(mae, mse, rmse, mape, mspe))
        f.write('\n')
        f.close()
