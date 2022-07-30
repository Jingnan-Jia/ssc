# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# log_dict is used to record super parameters and metrics

import sys
sys.path.append("..")

import csv
import os
import threading
import time
from statistics import mean
from typing import Dict, Optional, Union
from medutils.medutils import count_parameters
import torch
import torch.nn as nn
import mlflow
from ssc_scoring.mymodules.inference import record_best_preds
from ssc_scoring.mymodules.mydata import LoadPos
from ssc_scoring.mymodules.myloss import get_loss
from ssc_scoring.mymodules.networks import get_net_pos, get_net_pos_enc
from ssc_scoring.mymodules.set_args_pos import get_args
from ssc_scoring.mymodules.tool import record_1st, record_2nd, record_cgpu_info, eval_net_mae, compute_metrics
from ssc_scoring.mymodules.path import PathPosInit, PathPos

# from kd_med import kd_loss, PreTrainedEnc, GetEncSConv
import kd_med
from mlflow import log_metric, log_metrics, log_param, log_params
from fvcore.nn import FlopCountAnalysis
FLOPs_done = False

def try_func(func):
    def _try_fun(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as err:
            print(err, file=sys.stderr)
            pass
    return _try_fun


log_metric = try_func(log_metric)
log_metrics = try_func(log_metrics)


def gpu_info(outfile):  # need to be in the main file because it will be executed by another thread
    gpu_name, gpu_usage, gpu_utis = record_cgpu_info(outfile)
    # log_dict['gpuname'], log_dict['gpu_mem_usage'], log_dict['gpu_util'] = gpu_name, gpu_usage, gpu_utis

    return None


def start_run(args, mode, net, enc_t, dataloader_dt, loss_fun, loss_fun_mae, opt, mypath, epoch_idx,
              valid_mae_best=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        scaler = torch.cuda.amp.GradScaler()
    else:
        device = torch.device("cpu")
        scaler = None
    print(mode + "ing ......")
    loss_path = mypath.loss(mode)
    if mode == 'train' or mode == 'validaug':
        net.train()
    else:
        net.eval()

    batch_idx = 0
    total_loss = 0
    total_loss_mae = 0

    t0 = time.time()
    # t_load_data, t_train_per_step = [], []
    data_idx = 0
    dataloader = dataloader_dt['dl']
    if mode=='train':
        dataset = dataloader_dt['ds']
    EPOCHS_RECORDING_TIME = 2
    for data in dataloader:
        # print('data_idx:', data_idx)
        data_idx += 1
        if epoch_idx < EPOCHS_RECORDING_TIME:  # only show first 3 epochs' data loading time
            t1 = time.time()
            log_metric('TLoad', t1 - t0, data_idx + epoch_idx * len(dataloader))

        batch_x = data['image_key'].to(device)
        batch_y = data['label_in_patch_key'].to(device)

        global FLOPs_done
        if not FLOPs_done:
            net.eval()
            net_flops = FlopCountAnalysis(net, batch_x)
            flops = net_flops.total()
            # print(f"net_flops_G: {flops/1000/1000/1000}")  # convert to G
            log_param('net_FLOPs_G', flops/1000/1000/1000)# convert to G
            FLOPs_done = True
            if mode == 'train':
                net.train()
            else:
                net.eval()

        if args.kd == 'dist':  # kd train
            no_cuda = not torch.cuda.is_available()
            loss = kd_med.kd_loss(batch_x, enc_t, net, no_cuda)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_mae = loss
        else:
            # print('level: ', data['level_key'])
            if args.level_node != 0:
                batch_level = data['level_key'].to(device)
                print('batch_level', batch_level.clone().cpu().numpy())
                batch_x = [batch_x, batch_level]
            sp_z = data['space_key'][:, 0].reshape(-1, 1).to(device)
            if device == torch.device('cuda'):
                with torch.cuda.amp.autocast():
                    if mode != 'train':
                        with torch.no_grad():
                            pred = net(batch_x)
                    else:
                        pred = net(batch_x)
                    # print('pred.shape', pred.size())
                    pred *= sp_z
                    batch_y *= sp_z

                    loss = loss_fun(pred, batch_y)
                    loss_mae = loss_fun_mae(pred, batch_y)
                if mode == 'train':  # update gradients only when training
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
            else:
                if mode != 'train':
                    with torch.no_grad():
                        pred = net(batch_x)
                else:
                    pred = net(batch_x)
                pred *= sp_z
                batch_y *= sp_z

                loss = loss_fun(pred, batch_y)
                loss_mae = loss_fun_mae(pred, batch_y)
                if mode == 'train':  # update gradients only when training
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            if args.kd == 'dist':
                print(f'loss: {loss.item()}')
            else:
                print('loss:', loss.item())
                print('pred:', (pred / sp_z).clone().detach().cpu().numpy())
                print('label:', (batch_y / sp_z).clone().detach().cpu().numpy())

        if epoch_idx < EPOCHS_RECORDING_TIME:
            t2 = time.time()
            log_metric('TUpdateWBatch', t2 - t1, data_idx + epoch_idx * len(dataloader))
            t0 = t2  # reset the t0

        total_loss += loss.item()
        total_loss_mae += loss_mae.item()
        batch_idx += 1

        # if 'gpuname' not in log_dict:
        #     p1 = threading.Thread(target=gpu_info, args=(args.outfile,))
        #     p1.start()
    # if mode=='train':
    #     dataset.update_cache()
    # t_load_data, t_train_per_step = mean(t_load_data), mean(t_train_per_step)
    # if "t_load_data" not in log_dict:
    #     log_metrics({"t_load_data": t_load_data, "t_train_per_step": t_train_per_step})
    # print({"t_load_data": t_load_data, "t_train_per_step": t_train_per_step})

    ave_loss = total_loss / batch_idx
    ave_loss_mae = total_loss_mae / batch_idx
    print("mode:", mode, "loss: ", ave_loss, "loss_mae: ", ave_loss_mae)
    log_metric(mode + 'LossEpoch', ave_loss, epoch_idx)
    log_metric(mode + 'MAEEpoch', ave_loss_mae, epoch_idx)


    if not os.path.isfile(loss_path):
        with open(loss_path, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['step', 'loss', 'mae'])
    with open(loss_path, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([epoch_idx, ave_loss, ave_loss_mae])

    if valid_mae_best is not None:
        if ave_loss_mae < valid_mae_best:
            print("old valid loss mae is: ", valid_mae_best)
            print("new valid loss mae is: ", ave_loss_mae)

            valid_mae_best = ave_loss_mae

            print('this model is the best one, save it. epoch id: ', epoch_idx)
            torch.save(net.state_dict(), mypath.model_fpath)
            torch.save(net, mypath.model_wt_structure_fpath)
            print('save_successfully at ', mypath.model_fpath)
        return valid_mae_best
    else:
        return None


# def get_kd_net(net_name: str, nb_cls: int) -> nn.Module:
#     if net_name == "med3d_resnet50":
#         net = med3d.resnet50(sample_input_W=args.z_size,
#                              sample_input_H=args.y_size,
#                              sample_input_D=args.x_size,
#                              shortcut_type='A',
#                              no_cuda=False,
#                              num_seg_classes=nb_cls)
#     elif net_name == "model_genesis":
#         net = None
#     else:
#         raise Exception("wrong kd net name")
#     return net


def train(args):
    id = args.id
    mypath = PathPos(id)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.train_on_level or args.level_node:
        outs = 1
    else:
        outs = 5
    if args.kd == 'dist':  # kd train
        no_cuda = not torch.cuda.is_available()
        enc_s: torch.nn.Module = get_net_pos_enc(name=args.net, nb_cls=outs, level_node=args.level_node)

        enc_t = kd_med.pre_trained_enc(args.kd_t_name, no_cuda)
        print(enc_t)
        print(enc_s)
        print(args.kd_t_name)
        print(args.net)
        batch_x = torch.ones((2, 1, args.z_size, args.x_size, args.y_size))
        net = kd_med.EncPlusConv(batch_x, enc_t, enc_s, 3, no_cuda).get()

    else:
        net: torch.nn.Module = get_net_pos(name=args.net, nb_cls=outs, level_node=args.level_node, pretrained=args.pretrained, base=args.base)
        enc_t = None
    print('net:', net)

    net_parameters = count_parameters(net)
    net_parameters = str(round(net_parameters / 1024 / 1024, 2))
    log_param('net_parameters_M', net_parameters)

    label_file = mypath.label_excel_fpath  # "dataset/SSc_DeepLearning/GohScores.xlsx"
    log_param('label_file', label_file)

    seed = 49
    log_param('data_shuffle_seed', seed)

    all_loader = LoadPos(args.resample_z, mypath, label_file, seed, args.fold, args.total_folds, args.ts_level_nb,
                         args.level_node,
                         args.train_on_level, args.z_size, args.y_size, args.x_size, args.batch_size, args.workers)
    # train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader = all_loader.load()
    data_dt = all_loader.load()

    net = net.to(device)
    if args.eval_id:
        valid_mae_best = eval_net_mae(mypath, PathPos(args.eval_id))
        net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))  # model_fpath need to exist
    else:
        valid_mae_best = 10000

    loss_fun = get_loss(args.loss)
    loss_fun_mae = nn.L1Loss()
    lr = 1e-4
    log_param('lr', lr)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)
    epochs = 0 if args.mode == 'infer' else args.epochs
    for i in range(epochs):  # 20000 epochs
        if args.mode in ['train', 'continue_train']:
            start_run(args, 'train', net, enc_t, data_dt['train'], loss_fun, loss_fun_mae, opt, mypath, i)
        if i % args.valid_period == 0:
            # run the validation
            valid_mae_best = start_run(args, 'valid', net, enc_t, data_dt['valid'], loss_fun, loss_fun_mae, opt, mypath, i,
                                       valid_mae_best)
            log_metric('valid_mae_best', valid_mae_best, i)
            start_run(args, 'validaug', net, enc_t, data_dt['validaug'], loss_fun, loss_fun_mae, opt, mypath, i)
            start_run(args, 'test', net, enc_t, data_dt['test'], loss_fun, loss_fun_mae, opt, mypath, i)

    # dataloader_dict = {'train': train_dataloader,
    #                    'valid': valid_dataloader,
    #                    'validaug': validaug_dataloader,
    #                    'test': test_dataloader}
    if args.kd != 'dist':
        record_best_preds(net, data_dt, mypath, args)
        log_dict = compute_metrics(mypath, PathPos(args.eval_id), {},
                                   modes=['train', 'valid', 'test', 'validaug'])
        log_params(log_dict)

    # data_dt['train']['ds'].shutdown()

    print('Finish all things!')


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://nodelogin02:5000")
    mlflow.set_experiment("ssc_scoring_pos")

    args = get_args()
    id = record_1st(PathPosInit().record_file)  # write super parameters from set_args.py to record file.

    with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
        args.id = id  # do not need to pass id seperately to the latter function
        tmp_args_dt = vars(args)
        tmp_args_dt['fold'] = 'all'
        log_params(tmp_args_dt)

        p1 = threading.Thread(target=record_cgpu_info, args=(args.outfile, ))
        p1.start()

        for fold in [1, 2, 3, 4]:
            id = record_1st(PathPosInit().record_file)  # write super parameters from set_args.py to record file.
            with mlflow.start_run(run_name=str(id) + '_fold_' + str(fold), tags={"mlflow.note.content": f"fold: {fold}"}, nested=True):
                args.fold = fold
                args.id = id  # do not need to pass id seperately to the latter function
                tmp_args_dt = vars(args)
                log_params(tmp_args_dt)

                train(args)
        # record_2nd('pos', current_id=id, log_dict=log_dict, args=args)  # write other parameters and metrics to record file.

        p1.do_run = False  # stop the thread
        p1.join()
