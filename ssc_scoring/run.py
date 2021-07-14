# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import copy
import csv
import os
import shutil
import time
from typing import (Optional, Union, Dict)

import matplotlib
import myutil.myutil as futil
import pandas as pd
import torch
import torch.nn as nn
import sys
sys.path.append("..")

# import streamlit as st
matplotlib.use('Agg')
from statistics import mean
import threading
from mymodules.set_args import args
from mymodules.tool import record_1st, record_2nd, record_GPU_info, compute_metrics
from mymodules.path import PathScoreInit
from mymodules.path import PathScore as Path
from mymodules.myloss import get_loss
from mymodules.networks.cnn_fc2d import get_net, ReconNet
from mymodules.mydata import LoadScore
from mymodules.inference import record_best_preds, round_to_5
from mymodules.path import PathPos, PathPosInit

LogType = Optional[Union[int, float, str]]  # int includes bool
log_dict: Dict[str, LogType] = {}  # a global dict to store variables saved to log files

def GPU_info(outfile):  # need to be in the main file because it will be executed by another thread
    gpu_name, gpu_usage, gpu_utis = record_GPU_info(outfile)
    log_dict['gpuname'], log_dict['gpu_mem_usage'], log_dict['gpu_util'] = gpu_name, gpu_usage, gpu_utis

    return None


def start_run(mode, net, dataloader, device, loss_fun, loss_fun_mae, opt, mypath, epoch_idx,
              valid_mae_best=None):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        scaler = torch.cuda.amp.GradScaler()
    else:
        device = torch.device("cpu")

    print(mode + "ing ......")
    loss_path = mypath.loss(mode)
    if mode == 'train' or mode == 'validaug':
        net.train()
    else:
        net.eval()

    batch_idx = 0
    total_loss = 0
    total_loss_mae = 0
    total_loss_mae_end5 = 0

    # with torch.profiler.record_function("training_function"):
    t0 = time.time()
    t_load_data, t_to_device, t_train_per_step = [], [], []
    for data in dataloader:
        # print('get data')
        if 'label_key' not in data:
            batch_x, batch_y = data['image_key'], data['image_key']
        else:
            batch_x, batch_y = data['image_key'], data['label_key']
            print('batch_y is: ')
            print(batch_y)
        t1 = time.time()
        t_load_data.append(t1 - t0)


        # with torch.profiler.record_function("to_device"):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        t2 = time.time()
        t_to_device.append(t2 - t1)

        print(f"batch_x.shape: {batch_x.shape}, batch_y.shape: {batch_y.shape} ")
        if args.r_c == "c":
            batch_y = batch_y.type(torch.LongTensor)  # crossentropy requires LongTensor
            batch_y = batch_y.to(device)
        if device == torch.device('cuda'):
            with torch.cuda.amp.autocast():
                if mode != 'train':
                    with torch.no_grad():
                        pred = net(batch_x)
                else:
                    pred = net(batch_x)

                loss = loss_fun(pred, batch_y)

                if args.r_c == "c":
                    pred = torch.argmax(pred, dim=1)
                    pred = pred.type(torch.FloatTensor)
                    pred = pred.to(device)
                    pred = pred * 5  # convert back to original scores
                    batch_y = batch_y * 5  # convert back to original scores
                    # pred = pred.type(torch.LongTensor)

                loss_mae = loss_fun_mae(pred, batch_y)
                pred_end5 = round_to_5(pred, device)
                loss_mae_end5 = loss_fun_mae(pred_end5, batch_y)
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

            loss = loss_fun(pred, batch_y)

            if args.r_c == "c":
                pred = torch.argmax(pred, dim=1)
                pred = pred.type(torch.FloatTensor)
                pred = pred.to(device)
                pred = pred * 5  # convert back to original scores
                batch_y = batch_y * 5  # convert back to original scores

            loss_mae = loss_fun_mae(pred, batch_y)
            pred_end5 = round_to_5(pred, device)
            loss_mae_end5 = loss_fun_mae(pred_end5, batch_y)

            if mode == 'train':  # update gradients only when training
                opt.zero_grad()
                loss.backward()
                opt.step()

        t3 = time.time()
        t_train_per_step.append(t3 - t2)

        print(f"loss: {loss.item()}, pred.shape: {pred.shape}")
        # with torch.profiler.record_function("average_loss"):

        total_loss += loss.item()
        total_loss_mae += loss_mae.item()
        total_loss_mae_end5 += loss_mae_end5.item()
        batch_idx += 1

        if 'gpuname' not in log_dict:
            p1 = threading.Thread(target=GPU_info, args=(args.outfile, ))
            p1.start()

        t0 = t3  # reset the t0

    if "t_load_data" not in log_dict:
        t_load_data, t_to_device, t_train_per_step = mean(t_load_data), mean(t_to_device), mean(t_train_per_step)
        log_dict.update({"t_load_data": t_load_data,
                         "t_to_device": t_to_device,
                         "t_train_per_step": t_train_per_step})

    ave_loss = total_loss / batch_idx
    ave_loss_mae = total_loss_mae / batch_idx
    ave_loss_mae_end5 = total_loss_mae_end5 / batch_idx
    print("mode:", mode, "loss: ", ave_loss, "loss_mae: ", ave_loss_mae, "loss_mae_end5: ", ave_loss_mae_end5)

    if not os.path.isfile(loss_path):
        with open(loss_path, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(['step', 'loss', 'mae', 'mae_end5'])
    with open(loss_path, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([epoch_idx, ave_loss, ave_loss_mae, ave_loss_mae_end5])

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


def get_mae_best(fpath):
    loss = pd.read_csv(fpath)
    mae = min(loss['mae'].to_list())
    return mae


def train(id_: int, log_dict):
    mypath = Path(id_)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = get_net(args.net, 3, args) if args.r_c == "r" else get_net(args.net, 21, args)
    net_parameters = futil.count_parameters(net)
    net_parameters = str(round(net_parameters / 1024 / 1024, 2))
    log_dict['net_parameters'] = net_parameters
    label_file = "dataset/SSc_DeepLearning/GohScores.xlsx"
    seed = 49
    log_dict['data_shuffle_seed'] = seed
    net = net.to(device)
    print('move net t device')

    loss_fun = get_loss(loss=args.loss)
    loss_fun_mae = nn.L1Loss()
    lr = 1e-4
    log_dict['lr'] = lr
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)


    if args.corse_pred_id!=0:
        mypath_pos = PathPos(args.corse_pred_id)
        mypath.corse_pred_dir = os.path.join(mypath_pos.id_dir, 'predicted_slices')
        # all_loader = LoadScore(mypath, label_file, seed, args)
        # all_load = all_loader.load(merge=args.corse_pred_id)
        # start_run('valid', net, all_load, device, loss_fun, loss_fun_mae, opt, mypath, 1000)
        # load_dt = {'valid': all_load}
        # record_best_preds(net, load_dt, mypath, args)
        # log_dict = compute_metrics(mypath, args.eval_id, log_dict)
        # print('Finish all things!')
        # return log_dict

    # else:
    all_loader = LoadScore(mypath, label_file, seed, args)
    train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader = all_loader.load()

    if args.eval_id:
        mypath2 = Path(args.eval_id)
        if args.mode == "transfer_learning":
            net_recon = ReconNet(net)
            net_recon.load_state_dict(torch.load(mypath2.model_fpath, map_location=torch.device("cpu")))
            net.features = copy.deepcopy(net_recon.features)  # only use the pretrained features
            del net_recon
            valid_mae_best = 10000

        elif args.mode in ["continue_train", "infer"]:
            shutil.copy(mypath2.model_fpath, mypath.model_fpath)  # make sure there is at least one model there
            for mo in ['train', 'valid', 'test']:
                shutil.copy(mypath2.loss(mo), mypath.loss(mo))  # make sure there is at least one model there
            try:
                shutil.copy(mypath2.loss('validaug'), mypath.loss('validaug'))
            except:
                pass
            net.load_state_dict(torch.load(mypath.model_fpath, map_location=torch.device("cpu")))
            valid_mae_best = get_mae_best(mypath2.loss('valid'))
            print(f'load model from {mypath2.model_fpath}, valid_mae_best is {valid_mae_best}')
        else:
            raise Exception("wrong mode: " + args.mode)
    else:
        valid_mae_best = 10000



    epochs = args.epochs

    # for i in range(epochs):  # 20000 epochs
    #     start_run('train', net, train_dataloader, device, loss_fun, loss_fun_mae, opt, mypath, i)
    #
    #     # run the validation
    #     if (i % args.valid_period == 0) or (i > epochs * 0.8):
    #         # with torch.profiler.record_function("valid_validaug_test"):
    #         valid_mae_best = start_run('valid', net, valid_dataloader, device, loss_fun, loss_fun_mae, opt,
    #                                    mypath, i, valid_mae_best)
    #         start_run('validaug', net, validaug_dataloader, device, loss_fun, loss_fun_mae, opt, mypath, i)
    #         start_run('test', net, test_dataloader, device, loss_fun, loss_fun_mae, opt, mypath, i)
    # print('start save trace')

    data_loaders = {'train': train_dataloader, 'valid': valid_dataloader, 'validaug': validaug_dataloader, 'test': test_dataloader}
    record_best_preds(net, data_loaders, mypath, args)
    log_dict = compute_metrics(mypath, Path(args.eval_id), log_dict)
    print('Finish all things!')
    return log_dict


if __name__ == "__main__":
    # set some global variables here, like log_dict, device, amp
    LogType = Optional[Union[int, float, str]]  # int includes bool
    LogDict = Dict[str, LogType]
    log_dict: LogDict = {}  # a global dict to store immutable variables saved to log files

    id: int = record_1st('score', args)  # write super parameters from set_args.py to record file.
    log_dict = train(id, log_dict)
    record_2nd('score', current_id=id, log_dict=log_dict, args=args)  # write other parameters and metrics to record file.
