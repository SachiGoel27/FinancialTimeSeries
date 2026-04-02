from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, financial_metrics, macro_metrics
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
import pdb

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x)
                else:
                    # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # path = os.path.join(self.args.checkpoints, setting)
        path = os.path.join('../unitsf_res/' + self.args.exp_name + '/checkpoints/', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        path_loss = os.path.join('../unitsf_res/' + self.args.exp_name + '/loss/', setting)
        if not os.path.exists(path_loss):
            os.makedirs(path_loss)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        loss_tracker = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            loss_tracker.append([train_loss, vali_loss, test_loss])

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        np.save(path_loss + '/loss.npy', loss_tracker)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        # Load training data for MASE calculation and benchmark computation
        train_data, train_loader = self._get_data(flag='train')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        last_obs_all = []  # Store last observed values before prediction horizon
        
        folder_path = '../unitsf_res/' + self.args.exp_name + '/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x)
                else:
                    # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                # Get last observed value (last value of input sequence) for financial metrics
                # This is used for price-direction metrics
                last_obs = batch_x[:, -1:, :].detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    last_obs_shape = last_obs.shape
                    last_obs = test_data.inverse_transform(last_obs.reshape(last_obs_shape[0] * last_obs_shape[1], -1)).reshape(last_obs_shape)
                last_obs = last_obs[:, :, f_dim:]
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                last_obs_all.append(last_obs)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        last_obs_all = np.concatenate(last_obs_all, axis=0)
        
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        last_obs_all = last_obs_all.reshape(-1, last_obs_all.shape[-2], last_obs_all.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = '../unitsf_res/' + self.args.exp_name + '/eval/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Compute standard metrics (backward compatible)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # =====================================================================
        # Financial Metrics Computation
        # =====================================================================
        
        # Get evaluation parameters from args
        eval_domain = getattr(self.args, 'eval_domain', 'return')
        eval_benchmark = getattr(self.args, 'eval_benchmark', 'auto')
        annualization = getattr(self.args, 'annualization', 252)
        
        # Prepare in-sample data for MASE calculation
        # Collect all training targets
        insample_targets = []
        for batch_x, batch_y, _, _ in train_loader:
            f_dim = -1 if self.args.features == 'MS' else 0
            batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:].numpy()
            insample_targets.append(batch_y_target)
        insample_targets = np.concatenate(insample_targets, axis=0).flatten()
        
        # Compute benchmark predictions based on eval_benchmark setting
        # Auto mode: select benchmark based on eval_domain
        if eval_benchmark == 'auto':
            if eval_domain == 'return':
                eval_benchmark = 'zero'
            elif eval_domain == 'price':
                eval_benchmark = 'last'
            elif eval_domain == 'macro':
                eval_benchmark = 'ar1'
            else:  # volatility
                eval_benchmark = 'mean'
        
        # Create benchmark prediction array
        if eval_benchmark == 'zero':
            benchmark_pred = np.zeros_like(preds)
        elif eval_benchmark == 'mean':
            train_mean = np.mean(insample_targets)
            benchmark_pred = np.full_like(preds, train_mean)
        elif eval_benchmark == 'last':
            # Broadcast last_obs to match prediction shape
            benchmark_pred = np.broadcast_to(last_obs_all, preds.shape).copy()
        elif eval_benchmark == 'ar1':
            # AR(1) benchmark will be computed in macro_metrics
            benchmark_pred = None
        else:
            benchmark_pred = np.zeros_like(preds)
        
        # Prepare last_obs for price metrics (broadcast to match pred shape)
        last_obs_broadcast = np.broadcast_to(last_obs_all, preds.shape).copy() if eval_domain == 'price' else None
        
        # Compute financial metrics (for non-macro domains)
        if eval_domain != 'macro':
            fin_metrics = financial_metrics(
                pred=preds,
                true=trues,
                target_type=eval_domain,
                insample=insample_targets,
                benchmark_pred=benchmark_pred,
                last_obs=last_obs_broadcast,
                annualization=annualization
            )
            
            # Add benchmark info to metrics
            fin_metrics['eval_benchmark_type'] = eval_benchmark
            
            print('\n=== Financial Metrics ===')
            for k, v in fin_metrics.items():
                if isinstance(v, float):
                    print(f'{k}: {v:.6f}')
                else:
                    print(f'{k}: {v}')
        else:
            fin_metrics = {}
        
        # =====================================================================
        # Macro Metrics Computation (for macro domain)
        # =====================================================================
        
        macro_metrics_dict = None
        if eval_domain == 'macro':
            # Prepare last_obs for macro metrics
            # Flatten last_obs_all for batch processing
            last_obs_flat = last_obs_all.reshape(last_obs_all.shape[0], -1).mean(axis=-1)
            
            # Compute macro metrics with AR(1) benchmark
            macro_metrics_dict = macro_metrics(
                pred=preds,
                true=trues,
                insample=insample_targets,
                last_obs_all=last_obs_flat
            )
            
            print('\n=== Macro Forecasting Metrics ===')
            print(f"AR(1) Coefficients: alpha={macro_metrics_dict['ar1_alpha']:.6f}, beta={macro_metrics_dict['ar1_beta']:.6f}")
            print(f"\n--- Overall Metrics ---")
            print(f"MSE: {macro_metrics_dict['mse']:.6f}")
            print(f"RMSE: {macro_metrics_dict['rmse']:.6f}")
            print(f"MAE: {macro_metrics_dict['mae']:.6f}")
            print(f"\n--- AR(1) Comparison ---")
            print(f"MSE Ratio (Model/AR1): {macro_metrics_dict['mse_ratio_ar1']:.6f}")
            print(f"RMSE Ratio (Model/AR1): {macro_metrics_dict['rmse_ratio_ar1']:.6f}")
            print(f"OOS R² vs AR(1): {macro_metrics_dict['oos_r2_ar1']:.6f}")
            print(f"Theil's U: {macro_metrics_dict['theil_u']:.6f}")
            print(f"\n--- AR(1) Baseline ---")
            print(f"MSE (AR1): {macro_metrics_dict['mse_ar1']:.6f}")
            print(f"RMSE (AR1): {macro_metrics_dict['rmse_ar1']:.6f}")
            
            print(f"\n--- RMSFE by Horizon ---")
            for h, rmsfe in macro_metrics_dict['rmsfe_by_horizon'].items():
                print(f"  {h}: {rmsfe:.6f}")
            
            print(f"\n--- Metrics by Horizon ---")
            for h, h_metrics in macro_metrics_dict['metrics_by_horizon'].items():
                print(f"  {h}:")
                print(f"    MSE: {h_metrics['mse']:.6f}, RMSE: {h_metrics['rmse']:.6f}, MAE: {h_metrics['mae']:.6f}")
                if 'mse_ratio_ar1' in h_metrics:
                    print(f"    MSE Ratio: {h_metrics['mse_ratio_ar1']:.6f}, OOS R² vs AR1: {h_metrics['oos_r2_ar1']:.6f}")
        
        # =====================================================================
        # Save Results
        # =====================================================================

        log_path = "./res/" + self.args.exp_name
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # Write standard metrics to log file
        f = open(log_path+"/result_long_term_forecast_{}.txt".format(self.args.seed), 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        
        # Write financial metrics to log file (if computed)
        if fin_metrics:
            f.write('=== Financial Metrics ===\n')
            for k, v in fin_metrics.items():
                if isinstance(v, float):
                    f.write(f'{k}: {v:.6f}\n')
                else:
                    f.write(f'{k}: {v}\n')
            f.write('\n')
        
        # Write macro metrics to log file (if computed)
        if macro_metrics_dict is not None:
            f.write('=== Macro Forecasting Metrics ===\n')
            f.write(f"AR(1) Coefficients: alpha={macro_metrics_dict['ar1_alpha']:.6f}, beta={macro_metrics_dict['ar1_beta']:.6f}\n")
            f.write(f"MSE: {macro_metrics_dict['mse']:.6f}\n")
            f.write(f"RMSE: {macro_metrics_dict['rmse']:.6f}\n")
            f.write(f"MAE: {macro_metrics_dict['mae']:.6f}\n")
            f.write(f"MSE Ratio (Model/AR1): {macro_metrics_dict['mse_ratio_ar1']:.6f}\n")
            f.write(f"RMSE Ratio (Model/AR1): {macro_metrics_dict['rmse_ratio_ar1']:.6f}\n")
            f.write(f"OOS R² vs AR(1): {macro_metrics_dict['oos_r2_ar1']:.6f}\n")
            f.write(f"Theil's U: {macro_metrics_dict['theil_u']:.6f}\n")
            f.write('\nRMSFE by Horizon:\n')
            for h, rmsfe in macro_metrics_dict['rmsfe_by_horizon'].items():
                f.write(f"  {h}: {rmsfe:.6f}\n")
            f.write('\n')
        
        f.close()

        # Save numpy arrays (backward compatible)
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'last_obs.npy', last_obs_all)
        
        # Save financial metrics as JSON (if computed)
        if fin_metrics:
            with open(folder_path + 'financial_metrics.json', 'w') as f:
                # Convert any numpy types to Python types for JSON serialization
                json_metrics = {}
                for k, v in fin_metrics.items():
                    if isinstance(v, (np.floating, np.integer)):
                        json_metrics[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        json_metrics[k] = v.tolist()
                    else:
                        json_metrics[k] = v
                json.dump(json_metrics, f, indent=2)
            print(f'\nFinancial metrics saved to: {folder_path}financial_metrics.json')
        
        # Save macro metrics as JSON (if computed)
        if macro_metrics_dict is not None:
            with open(folder_path + 'macro_metrics.json', 'w') as f:
                # Convert any numpy types to Python types for JSON serialization
                def convert_to_json_serializable(obj):
                    if isinstance(obj, (np.floating, np.integer)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_to_json_serializable(item) for item in obj]
                    else:
                        return obj
                
                json_macro = convert_to_json_serializable(macro_metrics_dict)
                json.dump(json_macro, f, indent=2)
            print(f'Macro metrics saved to: {folder_path}macro_metrics.json')

        return
