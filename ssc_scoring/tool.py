# -*- coding: utf-8 -*-
# @Time    : 7/5/21 5:23 PM
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
from path import Path

def add_best_metrics(df: pd.DataFrame, mypath: Path, index: int) -> pd.DataFrame:
    modes = ['train', 'validaug', 'valid']
    if args.if_test:
        modes.append('test')
    metrics_min = 'mae'
    df.at[index, 'metrics_min'] = 'mae'

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

            best_index = loss_df[metrics_min].idxmin()
            loss = loss_df['loss'][best_index]
            mae = loss_df[metrics_min][best_index]
        df.at[index, mode + '_loss'] = round(loss, 2)
        df.at[index, mode + '_mae'] = round(mae, 2)
    return df


def write_and_backup(df: pd.DataFrame, record_file: str, mypath: Path):
    df.to_csv(record_file, index=False)
    shutil.copy(record_file, 'cp_' + record_file)
    df_lastrow = df.iloc[[-1]]
    df_lastrow.to_csv(mypath.id_dir + '/' + record_file, index=False)  # save the record of the current ex


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


def record_1st(record_file, mode):
    lock = FileLock(record_file + ".lock")  # lock the file, avoid other processes write other things
    with lock:  # with this lock,  open a file for exclusive access
        with open(record_file, 'a') as csv_file:
            df, new_id = get_df_id(record_file)
            if mode=='train':
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

def record_2nd(record_file, current_id, log_dict):
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
        elapsed_time = time_diff(t1, t2)
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



def time_diff(t1: datetime, t2: datetime):
    # t1_date = datetime.datetime(t1.year, t1.month, t1.day, t1.hour, t1.minute, t1.second)
    # t2_date = datetime.datetime(t2.year, t2.month, t2.day, t2.hour, t2.minute, t2.second)
    t_elapsed = t2 - t1

    return str(t_elapsed).split('.')[0]  # drop out microseconds
