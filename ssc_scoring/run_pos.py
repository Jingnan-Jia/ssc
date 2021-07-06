# -*- coding: utf-8 -*-
# @Time    : 3/3/21 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import csv
import datetime
import glob
import os
import random
import shutil
import threading
import time
from statistics import mean
from typing import Callable, Dict, List, Optional, Sequence, Union, Tuple, Hashable, Mapping

import monai
import myutil.myutil as futil
import numpy as np
import nvidia_smi
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from filelock import FileLock
from monai.transforms import ScaleIntensityRange, RandGaussianNoise
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader

import confusion
import myresnet3d
from set_args_pos import args
from networks import med3d_resnet as med3d
from networks import get_net_pos

from mytrans import LoadDatad, MyNormalizeImagePosd, AddChannelPosd, RandomCropPosd, \
    RandGaussianNoise, CenterCropPosd, CropLevelRegiond, ComposePosd, CropCorseRegiond
from mydata import AllLoader
from path import Path
from tool import record_1st, record_2nd, record_GPU_info, eval_net_mae, DAS

from myloss import get_loss
from inference import record_best_preds


def GPU_info(outfile):  # need to be in the main file because it will be executed by another thread
    gpu_name, gpu_usage, gpu_utis = record_GPU_info(outfile)
    log_dict['gpuname'], log_dict['gpu_mem_usage'], log_dict['gpu_util'] = gpu_name, gpu_usage, gpu_utis

    return None


def start_run(mode, net, kd_net, dataloader, loss_fun, loss_fun_mae, opt, mypath, epoch_idx,
              valid_mae_best=None):
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
    t_load_data, t_to_device, t_train_per_step = [], [], []
    for data in dataloader:

        t1 = time.time()
        t_load_data.append(t1 - t0)

        batch_x = data['image_key'].to(device)
        # print('batch_x.shape', batch_x.size())
        batch_y = data['label_in_patch_key'].to(device)

        # print('level: ', data['level_key'])
        if args.level_node != 0:
            batch_level = data['level_key'].to(device)
            print('batch_level', batch_level.clone().cpu().numpy())
            batch_x = [batch_x, batch_level]
        sp_z = data['space_key'][:, 0].reshape(-1, 1).to(device)
        # print('sp_z', sp_z.clone().cpu().numpy())

        t2 = time.time()
        t_to_device.append(t2 - t1)
        # print(net)
        if amp:
            with torch.cuda.amp.autocast():
                if mode != 'train':
                    with torch.no_grad():
                        pred = net(batch_x)
                else:
                    pred = net(batch_x)
                print('pred.shape', pred.size())
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

        t3 = time.time()
        t_train_per_step.append(t3 - t2)

        print('loss:', loss.item(), 'pred:', (pred / sp_z).clone().detach().cpu().numpy(),
              'label:', (batch_y / sp_z).clone().detach().cpu().numpy())

        total_loss += loss.item()
        total_loss_mae += loss_mae.item()
        batch_idx += 1

        if 'gpuname' not in log_dict:
            p1 = threading.Thread(target=GPU_info)
            p1.start()

        t0 = t3  # reset the t0

    t_load_data, t_to_device, t_train_per_step = mean(t_load_data), mean(t_to_device), mean(t_train_per_step)
    if "t_load_data" not in log_dict:
        log_dict.update({"t_load_data": t_load_data,
                         "t_to_device": t_to_device,
                         "t_train_per_step": t_train_per_step})
    print({"t_load_data": t_load_data,
           "t_to_device": t_to_device,
           "t_train_per_step": t_train_per_step})

    ave_loss = total_loss / batch_idx
    ave_loss_mae = total_loss_mae / batch_idx
    print("mode:", mode, "loss: ", ave_loss, "loss_mae: ", ave_loss_mae)

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


def compute_metrics(mypath: Path):
    for mode in ['train', 'valid', 'test', 'validaug']:
        try:
            if args.eval_id:
                mypath2 = Path(args.eval_id)
                shutil.copy(mypath2.data(mode), mypath.data(mode))  # make sure there is at least one modedel there
                shutil.copy(mypath2.loss(mode), mypath.loss(mode))  # make sure there is at least one modedel there
                shutil.copy(mypath2.world(mode), mypath.world(mode))  # make sure there is at least one modedel there
                shutil.copy(mypath2.pred(mode), mypath.pred(mode))  # make sure there is at least one modedel there
                shutil.copy(mypath2.pred_int(mode),
                            mypath.pred_int(mode))  # make sure there is at least one modedel there
                shutil.copy(mypath2.pred_world(mode),
                            mypath.pred_world(mode))  # make sure there is at least one modedel there

            out_dt = confusion.confusion(mypath.world(mode), mypath.pred_world(mode), label_nb=args.z_size, space=1, cf_kp=False)
            log_dict.update(out_dt)

            icc_ = futil.icc(mypath.world(mode), mypath.pred_world(mode))
            log_dict.update(icc_)
        except FileNotFoundError:
            continue


def get_kd_net(net_name: str) -> nn.Module:
    if net_name == "med3d_resnet50":
        net = med3d.resnet50(sample_input_W=args.z_size,
                sample_input_H=args.y_size,
                sample_input_D=args.x_size,
                shortcut_type='A',
                no_cuda=False,
                num_seg_classes=5)
    elif net_name == "model_genesis":
        net = None
    return net


def train(id: int):
    mypath = Path(id)
    if args.train_on_level or args.level_node:
        outs = 1
    else:
        outs = 5
    net: torch.nn.Module = get_net_pos(name=args.net, nb_cls = outs, level_node = args.level_node)
    net_parameters = futil.count_parameters(net)
    net_parameters = str(round(net_parameters / 1024 / 1024, 2))  # convert to **.** M
    log_dict['net_parameters'] = net_parameters

    # data_dir = dataset_dir(args.resample_z)
    label_file = "dataset/SSc_DeepLearning/GohScores.xlsx"
    # log_dict['data_dir'] = data_dir
    log_dict['label_file'] = label_file
    log_dict['data_shuffle_seed'] = 49

    all_loader = AllLoader(args.resample_z, mypath, label_file, 49, args.fold, args.total_folds, args.ts_level_nb, args.level_node,
                 args.train_on_level, args.z_size, args.y_size, args.x_size, args.batch_size, args.workers)
    train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader = all_loader.load()
    net = net.to(device)
    if args.eval_id:
        valid_mae_best = eval_net_mae(args.eval_id, mypath)
        net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))  # model_fpath need to exist

    else:
        valid_mae_best = 10000

    loss_fun = get_loss(args.loss)
    loss_fun_mae = nn.L1Loss()
    lr = 1e-4
    log_dict['lr'] = lr
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay)
    epochs = 0 if args.mode == 'infer' else args.epochs
    if args.kd_net is not None:
        kd_net = get_kd_net(args.kd_net)
    else:
        kd_net = None
    for i in range(epochs):  # 20000 epochs
        if args.mode in ['train', 'continue_train']:
            start_run('train', net, kd_net, train_dataloader, loss_fun, loss_fun_mae, opt, mypath, i)
        if i % args.valid_period == 0:
            # run the validation
            valid_mae_best = start_run('valid', net, kd_net, valid_dataloader, loss_fun, loss_fun_mae, opt, mypath, i,
                                       valid_mae_best)
            start_run('validaug', net, kd_net, validaug_dataloader, loss_fun, loss_fun_mae, opt, mypath, i)
            if args.if_test:
                start_run('test', net, kd_net, test_dataloader, loss_fun, loss_fun_mae, opt, mypath, i)

    dataloader_dict = {'train': train_dataloader, 'valid': valid_dataloader, 'validaug': validaug_dataloader}
    if args.if_test:
        dataloader_dict.update({'test': test_dataloader})
    record_best_preds(net, dataloader_dict, mypath, device, amp)
    compute_metrics(mypath)
    print('Finish all things!')

#
# def SlidingLoader(fpath, world_pos, z_size, stride=1, batch_size=1, mode='valid'):
#     print(f'start load {fpath} for sliding window inference')
#     xforms = [LoadDatad(), MyNormalizeImagePosd()]
#
#         # xforms.append(CropLevelRegiond(args.level_node, args.train_on_level,
#         #                                height=args.z_size, rand_start=False, start=0))
#     trans = ComposePosd(xforms)
#
#
#     data = trans(data={'fpath_key': fpath, 'world_key': world_pos})
#
#     raw_x = data['image_key']
#     data['label_in_img_key'] = np.array(data['label_in_img_key'][args.train_on_level - 1])
#
#     label = data['label_in_img_key']
#     print('data_world_key', data['world_key'])
#
#     assert raw_x.shape[0] > z_size
#     start_lower: int = label - z_size
#     start_higher: int = label + z_size
#     start_lower = max(0, start_lower)
#     start_higher = min(raw_x.shape[0], start_higher)
#
#     # ranges = raw_x.shape[0] - z_size
#     print(f'ranges: {start_lower} to {start_higher}')
#
#     batch_patch = []
#     batch_new_label = []
#     batch_start = []
#     i = 0
#
#     start = start_lower
#     while start < label:
#         if i < batch_size:
#             print(f'start: {start}, i: {i}')
#             if args.infer_2nd:
#                 mypath2 = Path(args.eval_id)
#                 crop = CropCorseRegiond(level=args.train_on_level, height=args.z_size, start=start,
#                                         data_fpath=mypath2.data(mode), pred_world_fpath=mypath2.pred_world(mode))
#             else:
#                 crop = CropLevelRegiond(level_node=args.level_node, train_on_level=args.train_on_level, height=args.z_size, rand_start=False, start=start)
#             new_data = crop(data)
#             new_patch, new_label = new_data['image_key'], new_data['label_in_patch_key']
#             # patch: np.ndarray = raw_x[start:start + z_size]  # z, y, z
#             # patch = patch.astype(np.float32)
#             # new_label: torch.Tensor = label - start
#             new_patch = new_patch[None]  # add a channel
#             batch_patch.append(new_patch)
#             batch_new_label.append(new_label)
#             batch_start.append(start)
#
#             start += stride
#             i += 1
#
#         if start >= start_higher or i >= batch_size:
#             batch_patch = torch.tensor(np.array(batch_patch))
#             batch_new_label = torch.tensor(batch_new_label)
#             batch_start = torch.tensor(batch_start)
#
#             yield batch_patch, batch_new_label, batch_start
#
#             batch_patch = []
#             batch_new_label = []
#             batch_start = []
#             i = 0
#
#
# class Evaluater():
#     def __init__(self, net, dataloader, mode, mypath):
#         self.net = net
#         self.dataloader = dataloader
#         self.mode = mode
#         self.mypath = mypath
#     def run(self):
#         for batch_data in self.dataloader:
#             for idx in range(len(batch_data['image_key'])):
#                 print('len_batch', len(batch_data))
#                 print(batch_data['fpath_key'][idx], batch_data['ori_world_key'][idx])
#                 sliding_loader = SlidingLoader(batch_data['fpath_key'][idx], batch_data['ori_world_key'][idx],
#                                                z_size=args.z_size, stride=args.infer_stride, batch_size=args.batch_size,
#                                                mode=args.mode)
#                 pred_in_img_ls = []
#                 pred_in_patch_ls = []
#                 label_in_patch_ls = []
#                 for patch, new_label, start in sliding_loader:
#                     batch_x = patch.to(device)
#                     if args.level_node != 0:
#                         batch_level = torch.ones((len(batch_x), 1)) * args.train_on_level
#                         batch_level = batch_level.to(device)
#                         print('batch_level', batch_level.clone().cpu().numpy())
#                         batch_x = [batch_x, batch_level]
#
#                     if 'gpuname' not in log_dict:
#                         p1 = threading.Thread(target=record_GPU_info)
#                         p1.start()
#
#                     if amp:
#                         with torch.cuda.amp.autocast():
#                             with torch.no_grad():
#                                 pred = self.net(batch_x)
#                     else:
#                         with torch.no_grad():
#                             pred = self.net(batch_x)
#
#                     # pred = pred.cpu().detach().numpy()
#                     pred_in_patch = pred.cpu().detach().numpy()
#                     pred_in_patch_ls.append(pred_in_patch)
#
#                     start_np = start.numpy().reshape((-1, 1))
#                     pred_in_img = pred_in_patch + start_np  # re organize it to original coordinate
#                     pred_in_img_ls.append(pred_in_img)
#
#                     new_label_ = new_label + start_np
#                     label_in_patch_ls.append(new_label_)
#
#                 pred_in_img_all = np.concatenate(pred_in_img_ls, axis=0)
#                 pred_in_patch_all = np.concatenate(pred_in_patch_ls, axis=0)
#                 label_in_patch_all = np.concatenate(label_in_patch_ls, axis=0)
#
#                 batch_label: np.ndarray = batch_data['label_in_img_key'][idx].cpu().detach().numpy().astype(int)
#                 batch_preds_ave: np.ndarray = np.mean(pred_in_img_all, 0)
#                 batch_preds_int: np.ndarray = batch_preds_ave.astype(int)
#                 batch_preds_world: np.ndarray = batch_preds_ave * batch_data['space_key'][idx][0].item() + \
#                                                 batch_data['origin_key'][idx][0].item()
#                 batch_world: np.ndarray = batch_data['world_key'][idx].cpu().detach().numpy()
#                 head = ['L1', 'L2', 'L3', 'L4', 'L5']
#                 if args.train_on_level:
#                     head = [head[args.train_on_level - 1]]
#                 if idx < 5:
#                     futil.appendrows_to(self.mypath.pred(self.mode).split('.csv')[0] + '_' + str(idx) + '.csv',
#                                         pred_in_img_all, head=head)
#                     futil.appendrows_to(self.mypath.pred(self.mode).split('.csv')[0] + '_' + str(idx) + '_in_patch.csv',
#                                         pred_in_patch_all, head=head)
#                     futil.appendrows_to(
#                         self.mypath.label(self.mode).split('.csv')[0] + '_' + str(idx) + '_in_patch.csv',
#                         label_in_patch_all, head=head)
#
#                     pred_all_world = pred_in_img_all * batch_data['space_key'][idx][0].item() + \
#                                      batch_data['origin_key'][idx][0].item()
#                     futil.appendrows_to(self.mypath.pred(self.mode).split('.csv')[0] + '_' + str(idx) + '_world.csv',
#                                         pred_all_world, head=head)
#
#                 if args.train_on_level:
#                     batch_label = np.array(batch_label).reshape(-1, )
#                     batch_preds_ave = np.array(batch_preds_ave).reshape(-1, )
#                     batch_preds_int = np.array(batch_preds_int).reshape(-1, )
#                     batch_preds_world = np.array(batch_preds_world).reshape(-1, )
#                     batch_world = np.array(batch_world).reshape(-1, )
#                 futil.appendrows_to(self.mypath.label(self.mode), batch_label, head=head)  # label in image
#                 futil.appendrows_to(self.mypath.pred(self.mode), batch_preds_ave, head=head)  # pred in image
#                 futil.appendrows_to(self.mypath.pred_int(self.mode), batch_preds_int, head=head)
#                 futil.appendrows_to(self.mypath.pred_world(self.mode), batch_preds_world, head=head)  # pred in world
#                 futil.appendrows_to(self.mypath.world(self.mode), batch_world, head=head)  # 33 label in world
#
#
# def record_best_preds(net: torch.nn.Module, dataloader_dict: Dict[str, DataLoader], mypath: Path):
#     net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))  # load the best weights to do evaluation
#     net.eval()
#     for mode, dataloader in dataloader_dict.items():
#         evaluater = Evaluater(net, dataloader, mode, mypath, device, amp)
#         evaluater.run()
#         # except:
#         #     continue


if __name__ == "__main__":
    # set some global variables here, like log_dict, device, amp
    LogType = Optional[Union[int, float, str]]  # int includes bool
    log_dict: Dict[str, LogType] = {}  # a global dict to store immutable variables saved to log files

    if torch.cuda.is_available():  # set device and amp
        device = torch.device("cuda")
        amp = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        device = torch.device("cpu")
        amp = False
        scaler = None
    das = DAS(device, amp, scaler)

    log_dict['amp'] = das.amp

    record_file: str = 'records_pos.csv'
    id: int = record_1st(record_file, mode=args.mode) # write super parameters from set_args.py to record file.
    train(id)
    record_2nd(record_file, current_id=id, log_dict=log_dict)  # write other parameters and metrics to record file.
