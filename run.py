import argparse
import os
import torch
import torch.backends
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_imputation import Exp_Imputation
from utils.print_args import print_args
import random
import numpy as np


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    fix_seed = 2026
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Anomaly Detection & Imputation')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='anomaly_detection', help='task name, options:[imputation, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='SHADE', help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='Humidity_Anomalous', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/Humidity', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Humidity_01234', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=12, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=5, help='prior anomaly ratio (%%)')
    parser.add_argument('--interpolate', type=int, default=0, help='use interpolation or not before training')
    parser.add_argument('--dmf_imputation_test', type=int, default=0, help='use dmf imputation or not when testing')
    parser.add_argument('--dl_imputation_test', type=int, default=0, help='use deep learning imputation or not when testing')
    parser.add_argument('--imputation_model', type=str, default='DLinear', help='imputation model for testing')
    parser.add_argument('--select_threshold', type=str, default=None, help='threshold selection method,, options:[IQR, POT, MEAN_STD, MAD]')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    # normalization
    parser.add_argument('--affine', type=bool, default=True, help='whether to use learnable affine parameters or not')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    # MOE
    parser.add_argument('--num_experts', type=int, default=4, help='number of experts')
    parser.add_argument('--loss_coef', type=float, default=0.1, help='loss coefficient for cv importance')
    parser.add_argument('--num_nodes', type=int, default=32)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')
    parser.add_argument('--num_experts_list', nargs='+', type=int, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2])
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--batch_norm', type=int, default=0)

    # Ada_HGCN
    parser.add_argument('--hyperedge_num', type=int, help='options: [10, 20, 30, 50]')
    parser.add_argument('--k_hyperedge', type=int, default=3, help='options: [1, 3, 5, 10, 15, 20]')
    parser.add_argument('--loss_coef1', type=float, default=0.1, help='loss coefficient for node & hyper constraints')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_sl{}_dm{}_el{}_dl{}_df{}_ar{}_mr{}_ip{}_fc{}_eb{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.d_model,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.anomaly_ratio,
                args.mask_rate,
                args.interpolate,
                args.factor,
                args.embed,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            torch.manual_seed(fix_seed)
            np.random.seed(fix_seed)
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_dm{}_el{}_dl{}_df{}_ar{}_mr{}_ip{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.d_model,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.anomaly_ratio,
            args.mask_rate,
            args.interpolate,
            args.factor,
            args.embed,
            args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
