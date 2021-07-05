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
from torch.utils.data import WeightedRandomSampler

import confusion
import myresnet3d
from set_args_pos import args
from networks import med3d_resnet as med3d
from networks import get_net_pos

from mytrans import LoadDatad, MyNormalizeImagePosd, AddChannelPosd, RandomCropPosd, \
    RandGaussianNoise, CenterCropPosd, CropLevelRegiond, ComposePosd
from mydata import AllLoader


class DatasetPos(Dataset):
    """SSc scoring dataset."""

    def __init__(self, data: Sequence, xform: Union[Sequence[Callable], Callable] = None):
        self.data = data
        self.transform = xform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.transform:
            data = self.transform(self.data[idx])
        else:
            data = self.data[idx]

        data['image_key'] = torch.as_tensor(data['image_key'])
        data['label_in_patch_key'] = torch.as_tensor(data['label_in_patch_key'])

        return data


class Path:
    def __init__(self, id, check_id_dir=False) -> None:
        self.id = id  # type: int
        self.slurmlog_dir = 'slurmlogs'
        self.model_dir = 'models_pos'
        self.data_dir = 'dataset'

        self.id_dir = os.path.join(self.model_dir, str(int(id)))  # +'_fold_' + str(args.fold)
        if args.mode == 'train' and check_id_dir:  # when infer, do not check
            if os.path.isdir(self.id_dir):  # the dir for this id already exist
                raise Exception('The same id_dir already exists', self.id_dir)

        for dir in [self.slurmlog_dir, self.model_dir, self.data_dir, self.id_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
                print('successfully create directory:', dir)

        self.model_fpath = os.path.join(self.id_dir, 'model.pt')
        self.model_wt_structure_fpath = os.path.join(self.id_dir, 'model_wt_structure.pt')

    def label(self, mode: str):
        return os.path.join(self.id_dir, mode + '_label.csv')

    def pred(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred.csv')

    def pred_int(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_int.csv')

    def pred_world(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_world.csv')

    def world(self, mode: str):
        return os.path.join(self.id_dir, mode + '_world.csv')

    def loss(self, mode: str):
        return os.path.join(self.id_dir, mode + '_loss.csv')

    def data(self, mode: str):
        return os.path.join(self.id_dir, mode + '_data.csv')

    def dataset_dir(self, resample_z: int) -> str:
        if resample_z == 0:  # use original images
            res_dir: str = 'SSc_DeepLearning'
        elif resample_z == 256:
            res_dir = 'LowResolution_fix_size'
        elif resample_z == 512:
            res_dir = 'LowRes512_192_192'
        elif resample_z == 800:
            res_dir = 'LowRes800_160_160'
        elif resample_z == 1024:
            res_dir = 'LowRes1024_256_256'
        else:
            raise Exception("wrong resample_z:" + str(args.resample_z))
        return os.path.join(self.data_dir, res_dir)


def _bytes_to_megabytes(value_bytes):
    return round((value_bytes / 1024) / 1024, 2)


def record_mem_info():
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    print('int(memusage.strip())')

    return int(memusage.strip())


def record_cpu_info():
    pass


def record_GPU_info():
    if args.outfile:
        jobid_gpuid = args.outfile.split('-')[-1]
        tmp_split = jobid_gpuid.split('_')[-1]
        if len(tmp_split) == 2:
            gpuid = tmp_split[-1]
        else:
            gpuid = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
        gpuname = nvidia_smi.nvmlDeviceGetName(handle)
        gpuname = gpuname.decode("utf-8")
        log_dict['gpuname'] = gpuname
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_usage = str(_bytes_to_megabytes(info.used)) + '/' + str(_bytes_to_megabytes(info.total)) + ' MB'
        log_dict['gpu_mem_usage'] = gpu_mem_usage
        gpu_util = 0
        for i in range(5):
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            gpu_util += res.gpu
            time.sleep(1)
        gpu_util = gpu_util / 5
        log_dict['gpu_util'] = str(gpu_util) + '%'
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
    gpu_flag = True
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

        if gpu_flag:
            p1 = threading.Thread(target=record_GPU_info)
            p1.start()
            gpu_flag = False

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


def get_column(n, tr_y):
    column = [i[n] for i in tr_y]
    column = [j / 5 for j in column]  # convert labels from [0,5,10, ..., 100] to [0, 1, 2, ..., 20]
    return column


class MSEHigher(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):

        if torch.sum(y_pred) > torch.sum(y_true):
            loss = self.mse(y_pred, y_true)
            print('mormal loss')
        else:
            loss = self.mse(y_pred, y_true) * 5
            print("higher loss")

        return loss


class MsePlusMae(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, y_pred, y_true):
        mse = self.mse(y_pred, y_true)
        mae = self.mae(y_pred, y_true)
        print(f"mse loss: {mse}, mae loss: {mae}")
        return mse + mae


def get_mae_best(fpath):
    loss = pd.read_csv(fpath)
    mae = min(loss['mae'].to_list())
    return mae


def sampler_by_disext(tr_y):
    disext_list = []
    for sample in tr_y:
        if type(sample) in [list, np.ndarray]:
            disext_list.append(sample[0])
        else:
            disext_list.append(sample)
    disext_np = np.array(disext_list)
    disext_unique = np.unique(disext_np)
    class_sample_count = np.array([len(np.where(disext_np == t)[0]) for t in disext_unique])
    weight = 1. / class_sample_count
    disext_unique_list = list(disext_unique)
    samples_weight = np.array([weight[disext_unique_list.index(t)] for t in disext_np])

    # weight = [nb_nonzero/len(world_list) if e[0] == 0 else nb_zero/len(world_list) for e in world_list]
    samples_weight = samples_weight.astype(np.float32)
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_loss(args):
    if args.loss == 'mae':
        loss_fun = nn.L1Loss()
    elif args.loss == 'smooth_mae':
        loss_fun = nn.SmoothL1Loss()
    elif args.loss == 'mse':
        loss_fun = nn.MSELoss()
    elif args.loss == 'mse+mae':
        loss_fun = nn.MSELoss() + nn.L1Loss()  # for regression task
    elif args.loss == 'msehigher':
        loss_fun = MSEHigher()
    else:
        raise Exception("loss function is not correct ", args.loss)
    return loss_fun


def eval_net_mae(eval_id: int, net: torch.nn.Module, mypath: Path):
    mypath2 = Path(eval_id)
    shutil.copy(mypath2.model_fpath, mypath.model_fpath)  # make sure there is at least one model there
    for mo in ['train', 'validaug', 'valid', 'test']:
        try:
            shutil.copy(mypath2.loss(mo), mypath.loss(mo))  # make sure there is at least one model
        except FileNotFoundError:
            pass

    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))
    valid_mae_best = get_mae_best(mypath2.loss('valid'))
    print(f'load model from {mypath2.model_fpath}, valid_mae_best is {valid_mae_best}')
    return net, valid_mae_best


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

            out_dt = confusion.confusion(mypath.world(mode), mypath.pred_world(mode), label_nb=args.z_size, space=1)
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

    all_loader = AllLoader(mypath, label_file, 49, args.fold, args.total_folds, args.ts_level_nb, args.level_node,
                 args.train_on_level, args.z_size, args.y_size, args.x_size, args.batch_size, args.workers)
    train_dataloader, validaug_dataloader, valid_dataloader, test_dataloader = all_loader.load()
    net = net.to(device)
    if args.eval_id:
        net, valid_mae_best = eval_net_mae(args.eval_id, net, mypath)
    else:
        valid_mae_best = 10000

    loss_fun = get_loss(args)
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
    record_best_preds(net, dataloader_dict, mypath)
    compute_metrics(mypath)
    print('Finish all things!')


def SlidingLoader(fpath, world_pos, z_size, stride=1, batch_size=1):
    print(f'start load {fpath} for sliding window inference')
    xforms = [LoadDatad(), MyNormalizeImagePosd()]

        # xforms.append(CropLevelRegiond(args.level_node, args.train_on_level,
        #                                height=args.z_size, rand_start=False, start=0))
    trans = ComposePosd(xforms)


    data = trans(data={'fpath_key': fpath, 'world_key': world_pos})

    raw_x = data['image_key']
    label = data['label_in_img_key']

    assert raw_x.shape[0] > z_size
    start_lower: int = label - z_size
    start_higher: int = label + z_size
    start_lower = max(0, start_lower)
    start_higher = min(raw_x.shape[0], start_higher)

    # ranges = raw_x.shape[0] - z_size
    print(f'ranges: {start_lower} to {start_higher}')

    batch_patch = []
    batch_new_label = []
    batch_start = []
    i = 0

    start = start_lower
    while start < label:
        if i < batch_size:
            print(f'start: {start}, i: {i}')
            crop = CropLevelRegiond(level_node=args.level_node, train_on_level=args.train_on_level, height=args.z_size, rand_start=False, start=start)
            new_data = crop(data)
            new_patch, new_label = new_data['image_key'], new_data['label_in_patch_key']
            # patch: np.ndarray = raw_x[start:start + z_size]  # z, y, z
            # patch = patch.astype(np.float32)
            # new_label: torch.Tensor = label - start
            new_patch = new_patch[None]  # add a channel
            batch_patch.append(new_patch)
            batch_new_label.append(new_label)
            batch_start.append(start)

            start += stride
            i += 1

        if start >= start_higher or i >= batch_size:
            batch_patch = torch.tensor(np.array(batch_patch))
            batch_new_label = torch.tensor(batch_new_label)
            batch_start = torch.tensor(batch_start)

            yield batch_patch, batch_new_label, batch_start

            batch_patch = []
            batch_new_label = []
            batch_start = []
            i = 0


class Evaluater():
    def __init__(self, net, dataloader, mode, mypath):
        self.net = net
        self.dataloader = dataloader
        self.mode = mode
        self.mypath = mypath
    # gpu_flag = True
    def run(self):
        for batch_data in self.dataloader:
            for idx in range(len(batch_data['image_key'])):
                sliding_loader = SlidingLoader(batch_data['fpath_key'][idx], batch_data['world_key'][idx],
                                               z_size=args.z_size, stride=args.infer_stride, batch_size=args.batch_size)
                pred_in_img_ls = []
                pred_in_patch_ls = []
                label_in_patch_ls = []
                for patch, new_label, start in sliding_loader:
                    batch_x = patch.to(device)

                    # if self.mode == 'train' and gpu_flag:
                    #     p1 = threading.Thread(target=record_GPU_info)
                    #     p1.start()
                    #     gpu_flag = False

                    if amp:
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                pred = self.net(batch_x)
                    else:
                        with torch.no_grad():
                            pred = self.net(batch_x)

                    # pred = pred.cpu().detach().numpy()
                    pred_in_patch = pred.cpu().detach().numpy()
                    pred_in_patch_ls.append(pred_in_patch)

                    start_np = start.numpy().reshape((-1, 1))
                    pred_in_img = pred_in_patch + start_np  # re organize it to original coordinate
                    pred_in_img_ls.append(pred_in_img)

                    new_label_ = new_label + start_np
                    label_in_patch_ls.append(new_label_)

                pred_in_img_all = np.concatenate(pred_in_img_ls, axis=0)
                pred_in_patch_all = np.concatenate(pred_in_patch_ls, axis=0)
                label_in_patch_all = np.concatenate(label_in_patch_ls, axis=0)

                batch_label: np.ndarray = batch_data['label_in_img_key'][idx].cpu().detach().numpy().astype('Int64')
                batch_preds_ave: np.ndarray = np.mean(pred_in_img_all, 0)
                batch_preds_int: np.ndarray = batch_preds_ave.astype('Int64')
                batch_preds_world: np.ndarray = batch_preds_ave * batch_data['space_key'][idx][0].item() + \
                                                batch_data['origin_key'][idx][0].item()
                batch_world: np.ndarray = batch_data['world_key'][idx].cpu().detach().numpy()
                head = ['L1', 'L2', 'L3', 'L4', 'L5']
                if args.train_on_level:
                    head = [head[args.train_on_level - 1]]
                if idx < 5:
                    futil.appendrows_to(self.mypath.pred(self.mode).split('.csv')[0] + '_' + str(idx) + '.csv',
                                        pred_in_img_all, head=head)
                    futil.appendrows_to(self.mypath.pred(self.mode).split('.csv')[0] + '_' + str(idx) + '_in_patch.csv',
                                        pred_in_patch_all, head=head)
                    futil.appendrows_to(
                        self.mypath.label(self.mode).split('.csv')[0] + '_' + str(idx) + '_in_patch.csv',
                        label_in_patch_all, head=head)

                    pred_all_world = pred_in_img_all * batch_data['space_key'][idx][0].item() + \
                                     batch_data['origin_key'][idx][0].item()
                    futil.appendrows_to(self.mypath.pred(self.mode).split('.csv')[0] + '_' + str(idx) + '_world.csv',
                                        pred_all_world, head=head)

                if args.train_on_level:
                    batch_label = np.array(batch_label).reshape(-1, )
                    batch_preds_ave = np.array(batch_preds_ave).reshape(-1, )
                    batch_preds_int = np.array(batch_preds_int).reshape(-1, )
                    batch_preds_world = np.array(batch_preds_world).reshape(-1, )
                    batch_world = np.array(batch_world).reshape(-1, )
                futil.appendrows_to(self.mypath.label(self.mode), batch_label, head=head)  # label in image
                futil.appendrows_to(self.mypath.pred(self.mode), batch_preds_ave, head=head)  # pred in image
                futil.appendrows_to(self.mypath.pred_int(self.mode), batch_preds_int, head=head)
                futil.appendrows_to(self.mypath.pred_world(self.mode), batch_preds_world, head=head)  # pred in world
                futil.appendrows_to(self.mypath.world(self.mode), batch_world, head=head)  # 33 label in world


def record_best_preds(net: torch.nn.Module, dataloader_dict: Dict[str, DataLoader], mypath: Path):
    net.load_state_dict(torch.load(mypath.model_fpath, map_location=device))  # load the best weights to do evaluation
    net.eval()
    for mode, dataloader in dataloader_dict.items():
        evaluater = Evaluater(net, dataloader, mode, mypath)
        evaluater.run()
        # except:
        #     continue


def record_preds(mode, batch_y, pred, mypath, data):
    batch_label = batch_y.cpu().detach().numpy().astype('Int64')
    batch_preds = pred.cpu().detach().numpy()
    batch_preds_int = batch_preds.astype('Int64')
    batch_preds_world = batch_preds_int * data['space_key'] + data['origin_key']

    futil.appendrows_to(mypath.label(mode), batch_label)
    futil.appendrows_to(mypath.pred(mode), batch_preds)
    futil.appendrows_to(mypath.pred_int(mode), batch_preds_int)
    futil.appendrows_to(mypath.pred_world(mode), batch_preds_world)


def fill_running(df: pd.DataFrame):
    for index, row in df.iterrows():
        if 'State' not in list(row.index) or row['State'] in [None, np.nan, 'RUNNING']:
            try:
                jobid = row['outfile'].split('-')[-1].split('_')[0]  # extract job id from outfile name
                seff = os.popen('seff ' + jobid)  # get job information
                for line in seff.readlines():
                    line = line.split(
                        ': ')  # must have space to be differentiated from time format 00:12:34
                    if len(line) == 2:
                        key, value = line
                        key = '_'.join(key.split(' '))  # change 'CPU utilized' to 'CPU_utilized'
                        value = value.split('\n')[0]
                        df.at[index, key] = value
            except:
                pass
    return df


def correct_type(df: pd.DataFrame):
    for column in df:
        ori_type = type(df[column].to_list()[-1])  # find the type of the last valuable in this column
        if ori_type is int:
            df[column] = df[column].astype('Int64')  # correct type
    return df


def get_df_id(record_file: str):
    if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
        new_id = 1
        df = pd.DataFrame()
    else:
        df = pd.read_csv(record_file)  # read the record file,
        last_id = df['ID'].to_list()[-1]  # find the last ID
        new_id = int(last_id) + 1
    return df, new_id


def record_1st(record_file, current_id):
    lock = FileLock(record_file + ".lock")  # lock the file, avoid other processes write other things
    with lock:  # with this lock,  open a file for exclusive access
        with open(record_file, 'a') as csv_file:
            df, new_id = get_df_id(record_file)
            mypath = Path(new_id, check_id_dir=True)  # to check if id_dir already exist

            start_date = datetime.date.today().strftime("%Y-%m-%d")
            start_time = datetime.datetime.now().time().strftime("%H:%M:%S")
            # start record by id, date,time row = [new_id, date, time, ]
            idatime = {'ID': new_id, 'start_date': start_date, 'start_time': start_time}
            args_dict = vars(args)
            idatime.update(args_dict)  # followed by super parameters
            if len(df) == 0:  # empty file
                df = pd.DataFrame([idatime])  # need a [] , or need to assign the index for df
            else:
                index = df.index.to_list()[-1]  # last index
                for key, value in idatime.items():  # write new line
                    df.at[index + 1, key] = value  #

            df = fill_running(df)  # fill the state information for other experiments
            df = correct_type(df)  # aviod annoying thing like: ID=1.00
            write_and_backup(df, record_file, mypath)
    return new_id


def add_best_metrics(df: pd.DataFrame, mypath: Path, index: int) -> pd.DataFrame:
    modes = ['train', 'validaug', 'valid']
    if args.if_test:
        modes.append('test')
    for mode in modes:
        lock2 = FileLock(mypath.loss(mode) + ".lock")
        # when evaluating/inference old models, those files would be copied to new the folder
        with lock2:
            try:
                loss_df = pd.read_csv(mypath.loss(mode))
            except FileNotFoundError:  # copy loss files from old directory to here
                mypath2 = Path(args.eval_id)
                shutil.copy(mypath2.loss(mode), mypath.loss(mode))
                try:
                    loss_df = pd.read_csv(mypath.loss(mode))
                except FileNotFoundError:  # still cannot find the loss file in old directory, pass this mode
                    continue

            best_index = loss_df['mae'].idxmin()
            log_dict['metrics_min'] = 'mae'
            loss = loss_df['loss'][best_index]
            mae = loss_df['mae'][best_index]
        df.at[index, mode + '_loss'] = round(loss, 2)
        df.at[index, mode + '_mae'] = round(mae, 2)
    return df


def write_and_backup(df: pd.DataFrame, record_file: str, mypath: Path):
    df.to_csv(record_file, index=False)
    shutil.copy(record_file, 'cp_' + record_file)
    df_lastrow = df.iloc[[-1]]
    df_lastrow.to_csv(mypath.id_dir + '/' + record_file, index=False)  # save the record of the current ex


def record_2nd(record_file, current_id):
    lock = FileLock(record_file + ".lock")
    with lock:  # with this lock,  open a file for exclusive access
        df = pd.read_csv(record_file)
        index = df.index[df['ID'] == current_id].to_list()
        if len(index) > 1:
            raise Exception("over 1 row has the same id", id)
        elif len(index) == 0:  # only one line,
            index = 0
        else:
            index = index[0]

        start_date = datetime.date.today().strftime("%Y-%m-%d")
        start_time = datetime.datetime.now().time().strftime("%H:%M:%S")
        df.at[index, 'end_date'] = start_date
        df.at[index, 'end_time'] = start_time

        # usage
        f = "%Y-%m-%d %H:%M:%S"
        t1 = datetime.datetime.strptime(df['start_date'][index] + ' ' + df['start_time'][index], f)
        t2 = datetime.datetime.strptime(df['end_date'][index] + ' ' + df['end_time'][index], f)
        elapsed_time = check_time_difference(t1, t2)
        df.at[index, 'elapsed_time'] = elapsed_time

        mypath = Path(current_id)  # evaluate old model
        df = add_best_metrics(df, mypath, index)

        for key, value in log_dict.items():  # convert numpy to str before writing all log_dict to csv file
            if type(value) in [np.ndarray, list]:
                str_v = ''
                for v in value:
                    str_v += str(v)
                    str_v += '_'
                value = str_v
            df.loc[index, key] = value
            if type(value) is int:
                df[key] = df[key].astype('Int64')

        for column in df:
            if type(df[column].to_list()[-1]) is int:
                df[column] = df[column].astype('Int64')  # correct type again, avoid None/1.00/NAN, etc.

        args_dict = vars(args)
        args_dict.update({'ID': current_id})
        for column in df:
            if column in args_dict.keys() and type(args_dict[column]) is int:
                df[column] = df[column].astype(float).astype('Int64')  # correct str to float and then int
        write_and_backup(df, record_file, mypath)


def record_experiment(record_file: str, current_id: Optional[int] = None):
    if current_id is None:  # before the experiment
        new_id = record_1st(record_file, current_id)
        return new_id
    else:  # at the end of this experiments, find the line of this id, and record the other information
        record_2nd(record_file, current_id)


def check_time_difference(t1: datetime, t2: datetime):
    t1_date = datetime.datetime(t1.year, t1.month, t1.day, t1.hour, t1.minute, t1.second)
    t2_date = datetime.datetime(t2.year, t2.month, t2.day, t2.hour, t2.minute, t2.second)
    t_elapsed = t2_date - t1_date

    return str(t_elapsed).split('.')[0]  # drop out microseconds


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
    log_dict['amp'] = amp
    print('device', device)

    record_file: str = 'records_pos.csv'
    id: int = record_experiment(record_file)  # write super parameters from set_args.py to record file.
    train(id)
    record_experiment(record_file, current_id=id)  # write other parameters and metrics to record file.
