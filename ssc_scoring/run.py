# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import copy
import csv
import os
import shutil
import sys
import time
from typing import (Optional, Union, Dict)

import matplotlib
from medutils.medutils import count_parameters
from mlflow import log_metric, log_param, log_params
import mlflow
import torch
import torch.nn as nn

sys.path.append("..")

# import streamlit as st
matplotlib.use('Agg')
from statistics import mean
import threading
from ssc_scoring.mymodules.set_args import get_args
from ssc_scoring.mymodules.tool import record_1st, record_2nd, record_gpu_info, compute_metrics, eval_net_mae, record_artifacts, record_cgpu_info
from ssc_scoring.mymodules.path import PathScore as Path
from ssc_scoring.mymodules.myloss import get_loss
from ssc_scoring.mymodules.networks.cnn_fc2d import get_net, ReconNet
from ssc_scoring.mymodules.mydata import LoadScore
from ssc_scoring.mymodules.inference import record_best_preds, round_to_5
from ssc_scoring.mymodules.path import PathPos
from argparse import Namespace

LogType = Optional[Union[int, float, str]]  # a global type to store immutable variables saved to log files


def gpu_info(outfile: str) -> None:
    """Get GPU usage information.

    This function needs to be in the main file because it will be executed by another thread.
    
    Args:
        outfile: The format of `outfile` is: slurm-[JOB_ID].out

    Returns:
        None. The GPU information will be saved to global variable `log_dict`.

    Example:

    >>> gpu_info('slurm-98234.out')

    """
    gpu_name, gpu_usage, gpu_utis = record_gpu_info(outfile)
    log_dict['gpuname'], log_dict['gpu_mem_usage'], log_dict['gpu_util'] = gpu_name, gpu_usage, gpu_utis

    return None





def start_run(args, mode, net, dataloader, loss_fun, loss_fun_mae, opt, mypath, epoch_idx,
              valid_mae_best=None):
    """Start run one step of training.

    Args:
        args: args instance
        mode: Chosen from 'train', 'valid', 'validaug', 'test'.
        net: Network
        dataloader: Iterator to generate a batch of data
        loss_fun: Loss instance
        loss_fun_mae: Mae instance
        opt: Optimizer including network parameters and learning rate
        mypath: My custom path instance
        epoch_idx: Idx of the current epoch
        valid_mae_best: Best valid mae

    Returns:
        valid_mae_best if mode is 'valid' else None

    """
    print(mode + "ing ......")

    if torch.cuda.is_available():  # Get device and scaler
        device = torch.device("cuda")
        scaler = torch.cuda.amp.GradScaler()
    else:
        device = torch.device("cpu")
        scaler = None

    loss_path = mypath.loss(mode)
    if mode == 'train' or mode == 'validaug':
        net.train()
    else:
        net.eval()

    batch_idx = 0
    total_loss = 0
    total_loss_mae = 0
    total_loss_mae_end5 = 0  # Goh score is ended by 5

    t0 = time.time()
    t_load_data, t_to_device, t_train_per_step = [], [], []  # time of loading data, to_device and train a step
    for ite, data in enumerate(dataloader):
        if 'label_key' not in data:
            batch_x, batch_y = data['image_key'], data['image_key']
        else:
            batch_x, batch_y = data['image_key'], data['label_key']
        batch_y = torch.round(batch_y / 25) * 25
        print(f'batch_y is: {batch_y}')
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
        if device == torch.device('cuda'):  # on GPU
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

        else:  # on CPU
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
        print(f'batch_pred is: {pred}')

        t3 = time.time()
        t_train_per_step.append(t3 - t2)

        print(f"loss: {loss.item()}, pred.shape: {pred.shape}")
        # with torch.profiler.record_function("average_loss"):
        loss_item = loss.item()
        loss_mae_item = loss_mae.item()
        # log_metric(mode+'LossBatch', loss_item, epoch_idx * len(dataloader) + ite)
        # log_metric(mode+'MAEBatch', loss_mae_item, epoch_idx * len(dataloader) + ite)

        total_loss += loss_item
        total_loss_mae += loss_mae_item
        total_loss_mae_end5 += loss_mae_end5.item()
        batch_idx += 1

        if 'gpuname' not in log_dict:
            p1 = threading.Thread(target=gpu_info, args=(args.outfile,))
            p1.start()

    if "t_load_data" not in log_dict:
        t_load_data, t_to_device, t_train_per_step = mean(t_load_data), mean(t_to_device), mean(t_train_per_step)
        log_dict.update({"t_load_data": t_load_data,
                         "t_to_device": t_to_device,
                         "t_train_per_step": t_train_per_step})

    ave_loss = total_loss / batch_idx
    ave_loss_mae = total_loss_mae / batch_idx
    ave_loss_mae_end5 = total_loss_mae_end5 / batch_idx
    log_metric(mode + 'LossEpoch', ave_loss, epoch_idx)
    log_metric(mode + 'MAEEpoch', ave_loss_mae, epoch_idx)

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


def train(args: Namespace, id_: int, log_dict: Dict[str, LogType]) -> Dict[str, LogType]:
    """The main body of the training process.

    Args:
        args: argparse instance
        id_: experiment ID
        log_dict: a dict to save super parameters and metrics

    Returns:
        log_dict

    """
    mypath = Path(id_)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = get_net(args.net, 3, args) if args.r_c == "r" else get_net(args.net, 21, args)  # 3 scores per image
    net_parameters = count_parameters(net)
    net_parameters = str(round(net_parameters / 1024 / 1024, 2))
    log_param('net_parameters_M', net_parameters)
    log_dict['net_parameters'] = net_parameters
    label_file = mypath.label_excel_fpath # "dataset/SSc_DeepLearning/GohScores.xlsx"  # labels are from here
    seed = 49  # for split of  cross-validation
    log_dict['data_shuffle_seed'] = seed
    log_param('data_shuffle_seed', seed)

    net = net.to(device)
    print('move net to device')

    loss_fun = get_loss(loss=args.loss)
    loss_fun_mae = nn.L1Loss()
    lr = 1e-4  # learning rate is fixed
    log_dict['lr'] = lr
    log_param('lr', lr)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)  # weight decay is L2 weight norm

    if args.eval_id:  # evaluate trained network
        mypath2 = Path(args.eval_id)
        if args.mode == "transfer_learning":  # todo: using other pre-trained weights apart from imageNet
            net_recon = ReconNet(net)
            net_recon.load_state_dict(torch.load(mypath2.model_fpath, map_location=torch.device("cpu")))
            net.features = copy.deepcopy(net_recon.features)  # only use the pretrained features
            del net_recon
            valid_mae_best = 10000

        elif args.mode in ["continue_train", "infer"]:  # Load old weights for continue_train or inference
            valid_mae_best = eval_net_mae(mypath, Path(args.eval_id))
            net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))  # model_fpath need to exist
        else:
            raise Exception("wrong mode: " + args.mode)
    else:
        valid_mae_best = 10000  # initiate mae_best as a very big value

    if args.corse_pred_id not in [0, None]:  # Load slices generated from PosNet, predict the scores of the slices
        mypath_pos = PathPos(args.corse_pred_id)  # provide 2D slice directory
        # mypath = Path(str(id_) + '_from_cascaded_slice')
        mypath.corse_pred_dir = os.path.join(mypath_pos.id_dir, 'predicted_slices')
        all_loader = LoadScore(mypath, label_file, seed, args)
        train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader = all_loader.load()
        # start_run('valid', net, all_load, device, loss_fun, loss_fun_mae, opt, mypath, 1000)
        load_dt = {'valid': valid_dataloader}
        record_best_preds(net, load_dt, mypath, args)
        log_dict = compute_metrics(mypath, args.eval_id, log_dict, modes='valid')
        print(f'Finish metrics on corse_pred_id: {args.corse_pred_id}!')
        return log_dict

    all_loader = LoadScore(mypath, label_file, seed, args)
    train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader = all_loader.load()

    epochs = args.epochs

    for i in range(epochs):  # loop epochs
        start_run(args, 'train', net, train_dataloader, loss_fun, loss_fun_mae, opt, mypath, i)

        # run the validation & testing
        if (i % args.valid_period == 0) or (i > epochs * 0.9):  # validation period become 1 at the end
            valid_mae_best = start_run(args, 'valid', net, valid_dataloader, loss_fun, loss_fun_mae, opt,
                                       mypath, i, valid_mae_best)
            log_metric('valid_mae_best', valid_mae_best, i)
            start_run(args, 'validaug', net, validaug_dataloader, loss_fun, loss_fun_mae, opt, mypath, i)
            start_run(args, 'test', net, test_dataloader, loss_fun, loss_fun_mae, opt, mypath, i)

    data_loaders = {'train': train_dataloader,
                    'valid': valid_dataloader,
                    'validaug': validaug_dataloader,
                    'test': test_dataloader}
    # Load the best model, get the corresponding prediction and metrics
    record_best_preds(net, data_loaders, mypath, args)
    tmp_dict = {}
    tmp_dict = compute_metrics(mypath, Path(args.eval_id), tmp_dict)
    log_params(tmp_dict)
    print('Finish all training/validation/testing + metrics!')
    log_dict.update(tmp_dict)
    return log_dict


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://nodelogin02:5000")
    mlflow.set_experiment("ssc_scoring")

    args = get_args()
    id: int = record_1st('score', args)  # write super parameters from set_args.py to record file.

    with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
        # p1 = threading.Thread(target=record_cgpu_info, args=(args.outfile,))
        # p1.start()

        log_params(vars(args))
        log_dict: Dict[str, LogType] = {}  # a global dict to store variables saved to log files

        log_param('ID', id)
        log_dict = train(args, id, log_dict)
        # log_params(log_dict)
        record_2nd('score', current_id=id, log_dict=log_dict, args=args)  # write more parameters & metrics to record file.

        # p1.do_run = False  # stop the thread
        # p1.join()
