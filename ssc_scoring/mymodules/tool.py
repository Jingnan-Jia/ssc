# -*- coding: utf-8 -*-
# @Time    : 7/5/21 5:23 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import argparse
import datetime
import os
import shutil
import time
from typing import Union, Tuple
from medutils.medutils import icc

import numpy as np
import nvidia_smi
import pandas as pd
import torch
from filelock import FileLock
from torch.utils.data import WeightedRandomSampler

from ssc_scoring.mymodules.confusion_test import confusion
from ssc_scoring.mymodules.path import PathScoreInit, PathPosInit, PathScore, PathPos
import threading
from mlflow import log_metric, log_param, start_run, end_run, log_params, log_artifact
import psutil

def sampler_by_disext(tr_y, sys_ratio=None) -> WeightedRandomSampler:
    """Balanced sampler according to score distribution of disext.

    Args:
        tr_y: Training labels.
            - Three scores per image: [[score1_disext, score1_gg, score1_ret], [score2_disext, score2_gg, score3_ret],
             ...]
            - One score per image: [score1_disext, score2_disext, ...]
        sys_ratio:

    Returns:
        WeightedRandomSampler

    Examples:
        :func:`ssc_scoring.mymodules.mydata.LoadScore.load`
    """
    disext_list = []
    for sample in tr_y:
        if type(sample) in [list, np.ndarray]:
            disext_list.append(sample[0])
        else:
            disext_list.append(sample)
    disext_np = np.array(disext_list)
    disext_unique = np.unique(disext_np)
    disext_unique_list = list(disext_unique)

    class_sample_count = np.array([len(np.where(disext_np == t)[0]) for t in disext_unique])
    if sys_ratio:
        weight = 1 / class_sample_count
        weight_sum = np.sum(weight)
        weight = np.array([w / weight_sum for w in weight])  # normalize the sum of weights to 1
        weight = (1 - sys_ratio) * weight  # scale the sume of weights to (1-sys_ratio)
        idx_0 = disext_unique_list.index(0)
        weight[idx_0] += sys_ratio
        sys_ratio_in_0 = sys_ratio / weight[idx_0]



        # weight[idx_0] += 20 * weight[idx_0]
        # samples_weight = np.array([weight[disext_unique_list.index(t)] for t in disext_np])
        #
        # weight_0 = sys_ratio + (1-sys_ratio)/21  # weight for category of 0, which is for original 0 and sys 0
        # weight_others = 1 - weight_0  # weight for other categories
        # # weight = [weight_0, *weight_others]
        # samples_weight = np.array([weight_0 if t==0 else weight_others for t in disext_np])
        # print("weight: ", weight)
        # print(samples_weight)
    else:
        weight = 1. / class_sample_count

    print("class_sample_count", class_sample_count)
    print("unique_disext", disext_unique_list)
    print("weight: ", weight)

    samples_weight = np.array([weight[disext_unique_list.index(t)] for t in disext_np])

    # weight = [nb_nonzero/len(data_y_list) if e[0] == 0 else nb_zero/len(data_y_list) for e in data_y_list]
    samples_weight = samples_weight.astype(np.float32)
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    print(list(sampler))
    if sys_ratio:
        return sampler, sys_ratio_in_0
    else:
        return sampler

def compute_metrics(mypath: Union[PathScore, PathPos],
                    mypath2: Union[PathScore, PathPos] = None,
                    log_dict: dict = None,
                    modes = None) -> dict:
    """Compute metrics and record them to log_dict.

    Args:
        mypath: Current experiment Path instance
        mypath2: Trained experiment Path instance, if mypath is empty, copy files from mypath2 to mypath
        log_dict: A dict to record all metrics

    Returns:
        log_dict

    Examples:
        :func:`ssc_scoring.run.train` and :func:`ssc_scoring.run_pos.train`

    """
    if modes is None:
        modes = ['train', 'valid', 'test', 'validaug']
    elif not isinstance(modes, list):
        modes = [modes]

    for mode in modes:
        if mypath.project_name == 'score':
            label = mypath.label(mode)
            pred = mypath.pred_end5(mode)
        else:
            label = mypath.world(mode)  # compare world metrics makes sure all experiments are compatible
            pred = mypath.pred_world(mode)
        try:
            if (not os.path.isfile(label)) and (mypath2 is not None):
                # mypath2 = Path(eval_id)
                shutil.copytree(mypath2.id_dir, mypath.id_dir, dirs_exist_ok=True)

            out_dt = confusion(label, pred)
            log_dict.update(out_dt)

            icc_ = icc(label, pred)
            log_dict.update(icc_)
        except FileNotFoundError:
            continue
    return log_dict


def get_mae_best(fpath: str) -> float:
    """Get minimum mae.

    Args:
        fpath: A csv file in which the `mae` at each epoch is recorded

    Returns:
        Minimum mae

    Examples:
        :func:`ssc_scoring.mymodules.tool.eval_net_mae`

    """

    loss = pd.read_csv(fpath)
    mae = min(loss['mae'].to_list())
    return mae


def eval_net_mae(mypath: Union[PathScore, PathPos], mypath2: Union[PathScore, PathPos]) -> float:
    """Copy trained model and loss log to new directory and get its valid_mae_best.

    Args:
        mypath: Current experiment Path instance
        mypath2: Trained experiment Path instance, if mypath is empty, copy files from mypath2 to mypath

    Returns:
        valid_mae_minimum

    Examples:
        :func:`ssc_scoring.run.train` and :func:`ssc_scoring.run_pos.train`

    """
    shutil.copy(mypath2.model_fpath, mypath.model_fpath)  # make sure there is at least one model there
    for mo in ['train', 'validaug', 'valid', 'test']:
        try:
            shutil.copy(mypath2.loss(mo), mypath.loss(mo))  # make sure there is at least one model
        except FileNotFoundError:
            pass
    valid_mae_best = get_mae_best(mypath2.loss('valid'))
    print(f'load model from {mypath2.model_fpath}, valid_mae_best is {valid_mae_best}')
    return valid_mae_best


def add_best_metrics(df: pd.DataFrame,
                     mypath: Union[PathScore, PathPos],
                     mypath2: Union[PathScore, PathPos],
                     index: int) -> pd.DataFrame:
    """Add best metrics: loss, mae (and mae_end5 if possible) to `df` in-place.

    Args:
        df: A DataFrame saving metrics (and other super-parameters)
        mypath: Current Path instance
        mypath2: Old Path instance, if the loss file can not be find in `mypath`, copy it from `mypath2`
        index: Which row the metrics should be writen in `df`

    Returns:
        `df`

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_2nd`

    """
    modes = ['train', 'validaug', 'valid', 'test']
    if mypath.project_name == 'score':
        metrics_min = 'mae_end5'
    else:
        metrics_min = 'mae'
    df.at[index, 'metrics_min'] = metrics_min

    for mode in modes:
        lock2 = FileLock(mypath.loss(mode) + ".lock")
        # when evaluating/inference old models, those files would be copied to new the folder
        with lock2:
            try:
                loss_df = pd.read_csv(mypath.loss(mode))
            except FileNotFoundError:  # copy loss files from old directory to here

                shutil.copy(mypath2.loss(mode), mypath.loss(mode))
                try:
                    loss_df = pd.read_csv(mypath.loss(mode))
                except FileNotFoundError:  # still cannot find the loss file in old directory, pass this mode
                    continue

            best_index = loss_df[metrics_min].idxmin()
            loss = loss_df['loss'][best_index]
            mae = loss_df['mae'][best_index]
            if mypath.project_name == 'score':
                mae_end5 = loss_df['mae_end5'][best_index]
                df.at[index, mode + '_mae_end5'] = round(mae_end5, 2)
        df.at[index, mode + '_loss'] = round(loss, 2)
        df.at[index, mode + '_mae'] = round(mae, 2)
    return df


def write_and_backup(df: pd.DataFrame, record_file: str, mypath: Union[PathScore, PathPos]) -> None:
    """Write `df` to `record_file` and backup it to `mypath`.

    Args:
        df: A DataFrame saving metrics (and other super-parameters)
        record_file: A file in hard disk saving df
        mypath: Path instance

    Returns:
        None. Results are saved to disk.

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_1st` and :func:`ssc_scoring.mymodules.tool.record_2nd`

    """
    df.to_csv(record_file, index=False)
    shutil.copy(record_file, os.path.join(mypath.results_dir, 'cp_' + os.path.basename(record_file)))
    df_lastrow = df.iloc[[-1]]
    df_lastrow.to_csv(os.path.join(mypath.id_dir, os.path.basename(record_file)),
                      index=False)  # save the record of the current ex


def fill_running(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the old record of completed experiments if the state of them are still 'running'.

    Args:
        df: A DataFrame saving metrics (and other super-parameters)

    Returns:
        df itself

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_1st`

    """
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


def correct_type(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the type of values in `df`. to avoid the ID or other int valuables become float number.

        Args:
            df: A DataFrame saving metrics (and other super-parameters)

        Returns:
            df itself

        Examples:
            :func:`ssc_scoring.mymodules.tool.record_1st`

        """
    for column in df:
        ori_type = type(df[column].to_list()[-1])  # find the type of the last valuable in this column
        if ori_type is int:
            df[column] = df[column].astype('Int64')  # correct type
    return df


def get_df_id(record_file: str) -> Tuple[pd.DataFrame, int]:
    """Get the current experiment ID. It equals to the latest experiment ID + 1.

    Args:
        record_file: A file to record experiments details (super-parameters and metrics).

    Returns:
        dataframe and new_id

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_1st`

    """
    if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
        new_id = 1
        df = pd.DataFrame()
    else:
        df = pd.read_csv(record_file)  # read the record file,
        last_id = df['ID'].to_list()[-1]  # find the last ID
        new_id = int(last_id) + 1
    return df, new_id


def record_1st(task: str, args: argparse.Namespace) -> int:
    """First record in this experiment.

    Args:
        task: 'score' or 'pos' for score and position prediction respectively.
        args: arguments.

    Returns:
        new_id

    Examples:
        :func:`ssc_scoring.run` and :func:`ssc_scoring.run_pos`

    """

    if task == 'score':
        record_file = PathScoreInit().record_file
        from ssc_scoring.mymodules.path import PathScore as Path
    else:
        record_file = PathPosInit().record_file
        from ssc_scoring.mymodules.path import PathPos as Path

    lock = FileLock(record_file + ".lock")  # lock the file, avoid other processes write other things
    with lock:  # with this lock,  open a file for exclusive access
        with open(record_file, 'a'):
            df, new_id = get_df_id(record_file)
            if args.mode == 'train':
                mypath = Path(new_id, check_id_dir=True)  # to check if id_dir already exist
            else:
                mypath = Path(new_id, check_id_dir=True)

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


def record_2nd(task: str, current_id: int, log_dict: dict, args: argparse.Namespace) -> None:
    """Second time to save logs.

    Args:
        task: 'score' or 'pos' for score and position prediction respectively.
        current_id: Current experiment ID
        log_dict: dict to save super-parameters and metrics
        args: arguments

    Returns:
        None. log_dict saved to disk.

    Examples:
        :func:`ssc_scoring.run` and :func:`ssc_scoring.run_pos`

    """

    if task == 'score':
        record_file = PathScoreInit().record_file
        from ssc_scoring.mymodules.path import PathScore as Path
    else:
        record_file = PathPosInit().record_file
        from ssc_scoring.mymodules.path import PathPos as Path
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
        df = add_best_metrics(df, mypath, Path(args.eval_id), index)

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


def time_diff(t1: datetime, t2: datetime) -> str:
    """Time difference.

    Args:
        t1: time 1
        t2: time 2

    Returns:
        Elapsed time

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_2nd`

    """
    # t1_date = datetime.datetime(t1.year, t1.month, t1.day, t1.hour, t1.minute, t1.second)
    # t2_date = datetime.datetime(t2.year, t2.month, t2.day, t2.hour, t2.minute, t2.second)
    t_elapsed = t2 - t1

    return str(t_elapsed).split('.')[0]  # drop out microseconds


def _bytes_to_megabytes(value_bytes: int) -> float:
    """Convert bytes to megabytes.

    Args:
        value_bytes: bytes number

    Returns:
        megabytes

    Examples:
        :func:`ssc_scoring.mymodules.tool.record_gpu_info`

    """
    return round((value_bytes / 1024) / 1024, 2)


def record_mem_info() -> int:
    """

    Returns:
        Memory usage in kB

    .. warning::

        This function is not tested. Please double check its code before using it.

    """

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    print('int(memusage.strip())')

    return int(memusage.strip())


def record_gpu_info(outfile) -> Tuple:
    """Record GPU information to `outfile`.

    Args:
        outfile: The format of `outfile` is: slurm-[JOB_ID].out

    Returns:
        gpu_name, gpu_usage, gpu_util

    Examples:

        >>> record_gpu_info('slurm-98234.out')

        or

        :func:`ssc_scoring.run.gpu_info` and :func:`ssc_scoring.run_pos.gpu_info`

    """

    if outfile:
        jobid_gpuid = outfile.split('-')[-1]
        tmp_split = jobid_gpuid.split('_')[-1]
        if len(tmp_split) == 2:
            gpuid = tmp_split[-1]
        else:
            gpuid = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
        gpuname = nvidia_smi.nvmlDeviceGetName(handle)
        gpuname = gpuname.decode("utf-8")
        # log_dict['gpuname'] = gpuname
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_usage = str(_bytes_to_megabytes(info.used)) + '/' + str(_bytes_to_megabytes(info.total)) + ' MB'
        # log_dict['gpu_mem_usage'] = gpu_mem_usage
        gpu_util = 0
        for i in range(5):
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            gpu_util += res.gpu
            time.sleep(1)
        gpu_util = gpu_util / 5
        # log_dict['gpu_util'] = str(gpu_util) + '%'
        return gpuname, gpu_mem_usage, str(gpu_util) + '%'
    else:
        print('outfile is None, can not show GPU memory info')
        return None, None, None


def record_cgpu_info(outfile) -> Tuple:
    """Record GPU information to `outfile`.

    Args:
        outfile: The format of `outfile` is: slurm-[JOB_ID].out

    Returns:
        gpu_name, gpu_usage, gpu_util

    Examples:

        >>> record_gpu_info('slurm-98234.out')

        or

        :func:`ssc_scoring.run.gpu_info` and :func:`ssc_scoring.run_pos.gpu_info`

    """
    t = threading.currentThread()
    t.do_run = True

    if outfile:
        cpu_count = psutil.cpu_count()
        log_param('cpu_count', cpu_count)

        pid = os.getpid()
        python_process = psutil.Process(pid)

        jobid_gpuid = outfile.split('-')[-1]
        tmp_split = jobid_gpuid.split('_')[-1]
        if len(tmp_split) == 2:
            gpuid = tmp_split[-1]
        else:
            gpuid = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
        gpuname = nvidia_smi.nvmlDeviceGetName(handle)
        gpuname = gpuname.decode("utf-8")
        log_param('gpuname', gpuname)
        # log_dict['gpuname'] = gpuname

        # log_dict['gpu_mem_usage'] = gpu_mem_usage
        # gpu_util = 0
        i = 0
        period = 2  # 2 seconds
        while i<60*20:  # stop signal passed from t, monitor 20 minutes
            if t.do_run:
                memoryUse = python_process.memory_info().rss / 2. ** 30  # memory use in GB...I think
                log_metric('cpu_mem_used_GB_in_process_rss', memoryUse, step=i)
                memoryUse = python_process.memory_info().vms / 2. ** 30  # memory use in GB...I think
                log_metric('cpu_mem_used_GB_in_process_vms', memoryUse, step=i)
                cpu_percent = psutil.cpu_percent()
                log_metric('cpu_util_used_percent', cpu_percent, step=i)
                # gpu_mem = dict(psutil.virtual_memory()._asdict())
                # log_params(gpu_mem)
                cpu_mem_used = psutil.virtual_memory().percent
                log_metric('cpu_mem_used_percent', cpu_mem_used, step=i)

                res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                # gpu_util += res.gpu
                log_metric("gpu_util", res.gpu, step=i)

                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                # gpu_mem_used = str(_bytes_to_megabytes(info.used)) + '/' + str(_bytes_to_megabytes(info.total))
                gpu_mem_used = _bytes_to_megabytes(info.used)
                log_metric('gpu_mem_used_MB', gpu_mem_used, step=i)

                time.sleep(period)
                i += period
            else:
                print('record_cgpu_info do_run is True, let stop the process')
                break
        print('It is time to stop this process: record_cgpu_info')
        return None
        # gpu_util = gpu_util / 5
        # gpu_mem_usage = str(gpu_mem_used) + ' MB'

        # log_dict['gpu_util'] = str(gpu_util) + '%'
        # return gpuname, gpu_mem_usage, str(gpu_util) + '%'


    else:
        print('outfile is None, can not show GPU memory info')
        return None, None, None


def record_artifacts(outfile):
    mythread = threading.currentThread()
    mythread.do_run = True
    if outfile:
        t = 0
        while 1:  # stop signal passed from t
            if mythread.do_run:
                log_artifact(outfile + '_err.txt')
                log_artifact(outfile + '_out.txt')
                if t <= 600:  # 10 minutes
                    period = 10
                    t += period
                else:
                    period = 60
                time.sleep(period)
            else:
                print('record_artifacts do_run is True, let stop the process')
                break

        print('It is time to stop this process: record_artifacts')
        return None
    else:
        print(f"No output file, no log artifacts")
        return None