import pandas as pd
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.exp_dmf import Exp_DMF
from exp.exp_dl import Exp_DL
from utils.interpolate import spatiotemporal_interpolation
from utils.select_threshold import iqr_threshold, pot_threshold, mean_std_threshold, mad_threshold
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, batch_x_time_mark, batch_x_space_mark, mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_time_mark = batch_x_time_mark.float().to(self.device)
                batch_x_space_mark = batch_x_space_mark.float().to(self.device)
                mask = mask.to(self.device)

                # random mask
                inp = batch_x.masked_fill(mask == 0, 0)

                # 插值
                if self.args.interpolate:
                    inp = spatiotemporal_interpolation(inp, mask)

                if self.args.model in ['SHADE']:
                    outputs, balance_loss = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)
                else:
                    outputs = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()

                if self.args.model in ['SHADE']:
                    balance_loss = balance_loss.detach().cpu()
                    loss = criterion(pred[mask == 1], true[mask == 1]) + balance_loss
                else:
                    loss = criterion(pred[mask == 1], true[mask == 1])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, _, batch_x_time_mark, batch_x_space_mark, mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_time_mark = batch_x_time_mark.float().to(self.device)
                batch_x_space_mark = batch_x_space_mark.float().to(self.device)
                mask = mask.to(self.device)

                # random mask
                inp = batch_x.masked_fill(mask == 0, 0)

                # 插值
                if self.args.interpolate:
                    inp = spatiotemporal_interpolation(inp, mask)

                if self.args.model in ['SHADE']:
                    outputs, balance_loss = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)
                else:
                    outputs = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]
                if self.args.model in ['SHADE']:
                    loss = criterion(outputs[mask == 1], batch_x[mask == 1]) + balance_loss
                else:
                    loss = criterion(outputs[mask == 1], batch_x[mask == 1])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')  # shuffle为True, 无法用于测试, 得设置shuffle为False重新构造dataloader
        train_loader = DataLoader(
            train_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False)
        num_train = train_data.num_train
        num_test = test_data.num_test
        train_mask = train_data.mask
        test_mask = test_data.mask

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        start_time = time.time()
        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, _, batch_x_time_mark, batch_x_space_mark, mask) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_time_mark = batch_x_time_mark.float().to(self.device)
                batch_x_space_mark = batch_x_space_mark.float().to(self.device)
                mask = mask.to(self.device)

                # random mask
                inp = batch_x.masked_fill(mask == 0, 0)

                # 插值
                if self.args.interpolate:
                    inp = spatiotemporal_interpolation(inp, mask)

                # reconstruction
                if self.args.model in ['SHADE']:
                    outputs, balance_loss = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)
                else:
                    outputs = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)

                # criterion
                score = self.anomaly_criterion(batch_x, outputs)
                score = torch.mean(score, dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0)
        train_energy = np.array(attens_energy)

        train_num_context = np.zeros((num_train, train_energy.shape[2]))
        train_rec_error = np.zeros((num_train, train_energy.shape[2]))

        for s in range(train_energy.shape[2]):
            num_context = 0
            for ts in range(num_train):
                if ts < self.args.seq_len - 1:
                    num_context = ts + 1
                elif ts >= self.args.seq_len - 1 and ts < num_train - self.args.seq_len + 1:
                    num_context = self.args.seq_len
                elif ts >= num_train - self.args.seq_len + 1:
                    num_context = num_train - ts
                train_num_context[ts][s] = num_context
        for t in range(len(train_energy)):
            train_rec_error[t:t + self.args.seq_len] += train_energy[t]
        avg_train_energy = (train_rec_error / train_num_context)
        unmasked_train_energy = avg_train_energy.T[train_mask.T == 1]

        # (2) find the threshold
        attens_energy = []
        for i, (batch_x, _, batch_x_time_mark, batch_x_space_mark, mask) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_x_time_mark = batch_x_time_mark.float().to(self.device)
            batch_x_space_mark = batch_x_space_mark.float().to(self.device)
            mask = mask.to(self.device)

            # random mask
            inp = batch_x.masked_fill(mask == 0, 0)

            # 插值
            if self.args.interpolate:
                inp = spatiotemporal_interpolation(inp, mask)

            # reconstruction
            if self.args.model in ['SHADE']:
                outputs, balance_loss = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)
            else:
                outputs = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)

            # criterion
            score = self.anomaly_criterion(batch_x, outputs)
            score = torch.mean(score, dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0)
        test_energy = np.array(attens_energy)

        test_num_context = np.zeros((num_test, test_energy.shape[2]))
        test_rec_error = np.zeros((num_test, test_energy.shape[2]))

        for s in range(test_energy.shape[2]):
            num_context = 0
            for ts in range(num_test):
                if ts < self.args.seq_len - 1:
                    num_context = ts + 1
                elif ts >= self.args.seq_len - 1 and ts < num_test - self.args.seq_len + 1:
                    num_context = self.args.seq_len
                elif ts >= num_test - self.args.seq_len + 1:
                    num_context = num_test - ts
                test_num_context[ts][s] = num_context
        for t in range(len(test_energy)):
            test_rec_error[t:t + self.args.seq_len] += test_energy[t]
        avg_test_energy = (test_rec_error / test_num_context)
        unmasked_test_energy = avg_test_energy.T[test_mask.T == 1]

        # 确定 threshold(只根据未被掩码的部分)
        combined_energy = np.concatenate([unmasked_train_energy, unmasked_test_energy], axis=0)

        if self.args.select_threshold:
            print(f'Use {self.args.select_threshold} method')
        else:
            print('Use percentile method')
        if self.args.select_threshold == 'IQR':
            threshold = iqr_threshold(combined_energy)
        elif self.args.select_threshold == 'POT':
            threshold = pot_threshold(combined_energy)
        elif self.args.select_threshold == 'MAD':
            threshold = mad_threshold(combined_energy)
        else:
            # 返回整个序列由大到小第 self.args.anomaly_ratio % 的得分值
            threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)

        print("Threshold :", threshold)
        print("Total cost time: {}".format(time.time() - start_time))

        # (3) evaluation on the test set
        print('########## Point Adjusted Evaluation ##########')
        pred = (avg_test_energy > threshold).astype(int)
        df_label = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path + '_label.csv'))
        cols_data = df_label.columns[2:]
        df_label = df_label[cols_data]
        test_labels = (df_label.values)[len(df_label) - num_test:]
        test_labels[test_mask == 0] = 0  # 被掩盖的部分视为正常点
        gt = test_labels.astype(int)
        # detection adjustment
        for s in range(test_labels.shape[1]):
            gt[:, s], pred[:, s] = adjustment(gt[:, s], pred[:, s])
        pred_copy = pred.copy()  # (num_test, S),用于DMF
        gt = gt[test_mask == 1]
        pred = pred[test_mask == 1]
        pred = np.array(pred)
        gt = np.array(gt)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Point Adjusted Evaluation  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        f.write('\n')
        f.close()


        # Completion
        if self.args.dmf_imputation_test:
            print('########## DMF Imputation Testing ##########')
            # print(test_mask.reshape(-1).size)
            # print((test_data.mask == 0).sum())
            test_origin_mask = test_mask.copy()
            test_mask[pred_copy == 1] = 0
            # print((test_data.mask == 0).sum())

            fix_seed = 2026
            torch.manual_seed(fix_seed)
            np.random.seed(fix_seed)
            exp = Exp_DMF(self.args)
            exp.test(original_mask=test_origin_mask, mask=test_mask, train_epochs=10000)  # 使用修改后的test_mask


        if self.args.dl_imputation_test:
            print('########## Deep Learning Imputation Testing ##########')
            test_origin_mask = test_mask.copy()
            test_mask[pred_copy == 1] = 0

            fix_seed = 2026
            torch.manual_seed(fix_seed)
            np.random.seed(fix_seed)
            exp = Exp_DL(self.args)
            exp.test(original_mask=test_origin_mask, mask=test_mask)  # 使用修改后的test_mask


        f = open("result_anomaly_detection.txt", 'a')
        f.write('\n')
        f.close()