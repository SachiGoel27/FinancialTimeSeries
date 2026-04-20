import argparse
import os
import sys
import torch
import random
import numpy as np

# =================================================================================
# =================================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
submodule_path = os.path.join(current_dir, 'TimeRecipe')
if os.path.exists(submodule_path):
    sys.path.append(submodule_path)
# =================================================================================

try:
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from utils.print_args import print_args
except ImportError as e:
    print(f"Error: Could not find model dependencies. Did you run 'git submodule update --init'?")
    print(f"Details: {e}")
    sys.exit(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='unitsf',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # Data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # Model define
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decomposition')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision', default=False)
    parser.add_argument('--des', type=str, default='test', help='exp description')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids')

    # Unitsf config
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--use_norm', type=str2bool, default=True, help='use instance normalization')
    parser.add_argument('--use_decomp', type=str2bool, default=False, help='use series decomposition')
    parser.add_argument('--fusion', type=str, default='temporal', help='temporal or feature fusion')
    parser.add_argument('--emb_type', type=str, default='token', help='embedding type')
    parser.add_argument('--ff_type', type=str, default='mlp', help='feed-forward type')
    parser.add_argument('--exp_name', type=str, default='test', help='setup name for saving')

    args = parser.parse_args()

    # =================================================================================
    # =================================================================================
    original_dataset_name = args.data
    custom_file_map = {
        'vix_futures': 'vix_futures.csv',
        'credit_spreads': 'credit_spreads.csv',
        'bitcoin': 'bitcoin.csv',
        'spy': 'spy.csv'
    }

    if args.data in custom_file_map:
        print(f"[Auto-Config] Detected custom dataset: {args.data}")
        args.root_path = './data/'
        args.data_path = custom_file_map[args.data]
        args.data = 'custom' 
    # =================================================================================

    args.use_gpu = True if torch.cuda.is_available() else False

    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast
    
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            
            setting = '{}_pl{}_sd{}_un{}_ud{}_fu{}_eb{}_ff{}_ft{}'.format(
                original_dataset_name, args.pred_len, fix_seed, args.use_norm,
                args.use_decomp, args.fusion, args.emb_type, args.ff_type, args.features)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        exp = Exp(args) 
        setting = '{}_pl{}_sd{}_un{}_ud{}_fu{}_eb{}_ff{}_ft{}'.format(
                original_dataset_name, args.pred_len, fix_seed, args.use_norm,
                args.use_decomp, args.fusion, args.emb_type, args.ff_type, args.features)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()