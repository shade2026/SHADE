from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
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


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

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

                outputs = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, :, f_dim:]
                mask = mask[:, :, :, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()

                loss = criterion(pred[mask == 0], true[mask == 0])
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

                outputs = self.model(inp, batch_x_time_mark, batch_x_space_mark, mask)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, :, f_dim:]
                mask = mask[:, :, :, f_dim:]

                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
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
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, batch_x_time_mark, batch_x_space_mark, mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_time_mark = batch_x_time_mark.float().to(self.device)
                batch_x_space_mark = batch_x_space_mark.float().to(self.device)
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

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print('mse:{}, mae:{}, rmse{}'.format(mse, mae, rmse))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse{}'.format(mse, mae, rmse))
        f.write('\n')
        f.write('\n')
        f.close()

        return
