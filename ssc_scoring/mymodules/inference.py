# -*- coding: utf-8 -*-
# @Time    : 7/6/21 7:17 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

from typing import (Union, Dict)

import monai
import medutils.medutils as futil

import numpy as np
import torch
from torch.utils.data import DataLoader

from ssc_scoring.mymodules.mytrans import LoadDatad, NormImgPosd, RandCropLevelRegiond, CropCorseRegiond, CropPosd
from ssc_scoring.mymodules.path import PathInit
from ssc_scoring.mymodules.path import PathPos as Path


def SlidingLoader(fpath, world_pos, z_size, stride=1, batch_size=1, mode='valid', args=None):
    print(f'start load {fpath} for sliding window inference')
    xforms = [LoadDatad(), NormImgPosd()]  #
    trans = monai.transforms.Compose(xforms)

    data = trans({'fpath_key': fpath, 'world_key': world_pos})
    raw_x = data['image_key']

    assert raw_x.shape[0] > z_size  # shape along z should be higher than z_size

    if args.train_on_level or args.level_node:  # the output 3D patch should be around the specific level
        data['label_in_img_key'] = np.array(data['label_in_img_key'][args.train_on_level - 1])
        label = data['label_in_img_key']
        start_lower: int = label - z_size  # start lower than this value can not crop a patch including the level
        start_lower = max(0, start_lower)  # if start_lower<0, use 0

        start_higher: int = label  # start higher than this value can not include the level
        start_higher = min(raw_x.shape[0] - z_size, start_higher) # if start_higher > raw_x.shape[0]- z_size
    else:  # if not level is assigned, start should range from 0 to raw_x.shape[0] - z_size
        start_lower = 0
        start_higher = raw_x.shape[0] - z_size
    # print('data_world_key', data['world_key'])

    # ranges = raw_x.shape[0] - z_size
    print(f'start point ranges: {start_lower} to {start_higher}')

    batch_patch = []
    batch_new_label = []
    batch_start = []
    i = 0

    start = start_lower
    while start < start_higher:
        if i < batch_size:
            print(f'start: {start}, i: {i}')
            if args.infer_2nd:
                mypath2 = Path(args.eval_id)
                crop = CropCorseRegiond(level_node=args.level_node,
                                        train_on_level=args.train_on_level,
                                        height=args.z_size,
                                        rand_start=False,
                                        start=start,
                                        data_fpath=mypath2.data(mode),
                                        pred_world_fpath=mypath2.pred_world(mode))
            elif args.level_node or args.train_on_level:
                crop = RandCropLevelRegiond(level_node=args.level_node,
                                            train_on_level=args.train_on_level,
                                            height=args.z_size,
                                            rand_start=False,
                                            start=start)
            else:
                crop = CropPosd(start=start, height=args.z_size)

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


def record_preds(mode, batch_y, pred, mypath):
    batch_label = batch_y.cpu().detach().numpy().astype('int')
    batch_preds = pred.cpu().detach().numpy()
    batch_preds_int = batch_preds.astype('int')
    batch_preds_end5 = round_to_5(batch_preds_int)
    batch_preds_end5 = batch_preds_end5.astype('int')

    head = ['disext', 'gg', 'retp']
    futil.appendrows_to(mypath.label(mode), batch_label, head=head)
    futil.appendrows_to(mypath.pred(mode), batch_preds, head=head)
    futil.appendrows_to(mypath.pred_int(mode), batch_preds_int, head=head)
    futil.appendrows_to(mypath.pred_end5(mode), batch_preds_end5, head=head)


def round_to_5(pred: Union[torch.Tensor, np.ndarray], device=torch.device("cpu")) -> Union[torch.Tensor, np.ndarray]:
    if type(pred) == torch.Tensor:
        tensor_flag = True
        pred = pred.cpu().detach().numpy()
    else:
        tensor_flag = False

    # elif type(pred) == np.ndarray:
    pred = np.rint(pred / 5) * 5
    pred[pred > 100] = 100
    pred[pred < 0] = 0

    if tensor_flag:
        pred = torch.tensor(pred)
        pred = pred.to(device)

    return pred


class Evaluater_pos():
    def __init__(self, net, dataloader, mode, mypath, args):
        self.net = net
        self.dataloader = dataloader
        self.mode = mode
        self.mypath = mypath
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = self.net.to(self.device).eval()
        self.amp = True if torch.cuda.is_available() else False
        self.args = args

    def run(self):
        for batch_data in self.dataloader:
            for idx in range(len(batch_data['image_key'])):
                print('len_batch', len(batch_data))
                print(batch_data['fpath_key'][idx], batch_data['ori_world_key'][idx])
                sliding_loader = SlidingLoader(batch_data['fpath_key'][idx], batch_data['ori_world_key'][idx],
                                               z_size=self.args.z_size, stride=self.args.infer_stride,
                                               batch_size=self.args.batch_size,
                                               mode=self.mode, args=self.args)
                pred_in_img_ls = []
                pred_in_patch_ls = []
                label_in_patch_ls = []
                for patch, new_label, start in sliding_loader:
                    batch_x = patch.to(self.device)
                    if self.args.level_node != 0:
                        batch_level = torch.ones((len(batch_x), 1)) * self.args.train_on_level
                        batch_level = batch_level.to(self.device)
                        print('batch_level', batch_level.clone().cpu().numpy())
                        batch_x = [batch_x, batch_level]

                    if self.amp:
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

                batch_label: np.ndarray = batch_data['label_in_img_key'][idx].cpu().detach().numpy().astype(int)
                batch_preds_ave: np.ndarray = np.median(pred_in_img_all, 0)  # todo: compare mean and medial!
                batch_preds_int: np.ndarray = batch_preds_ave.astype(int)
                batch_preds_world: np.ndarray = batch_preds_ave * batch_data['space_key'][idx][0].item() + \
                                                batch_data['origin_key'][idx][0].item()
                batch_world: np.ndarray = batch_data['world_key'][idx].cpu().detach().numpy()
                head = ['L1', 'L2', 'L3', 'L4', 'L5']
                if self.args.train_on_level:
                    head = [head[self.args.train_on_level - 1]]
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

                if self.args.train_on_level:
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


class Evaluater_score():
    def __init__(self, net, dataloader, mode, mypath, args):
        self.net = net
        self.dataloader = dataloader
        self.mode = mode
        self.mypath = mypath
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = self.net.to(self.device).eval()
        self.amp = True if torch.cuda.is_available() else False
        self.args = args

    def run(self):
        for data in self.dataloader:
            print(f'mode: {self.mode}, ==========')
            print(f"data from {data['fpath_key']}")

            if 'label_key' not in data:
                batch_x, batch_y = data['image_key'], data['image_key']
            else:
                batch_x, batch_y = data['image_key'], data['label_key']

            print('batch_y is: ')
            print(batch_y)

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            if self.amp:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        pred = self.net(batch_x)
            else:
                with torch.no_grad():
                    pred = self.net(batch_x)
            print(f'batch_pred is: {pred}')
            print(f'mode: {self.mode}, ==========')

            record_preds(self.mode, batch_y, pred, self.mypath)


def record_best_preds(net: torch.nn.Module, data_dict: Dict[str, DataLoader], mypath: Path, args):
    net.load_state_dict(torch.load(mypath.model_fpath))  # load the best weights to do evaluation
    for mode, data in data_dict.items():
        dataloader = data['dl'] if isinstance(data, dict) else data
        if mypath.project_name=='score':
            evaluater = Evaluater_score(net, dataloader, mode, mypath, args)
        else:
            evaluater = Evaluater_pos(net, dataloader, mode, mypath, args)
        evaluater.run()
