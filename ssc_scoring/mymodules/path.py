# -*- coding: utf-8 -*-
# @Time    : 7/5/21 7:24 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com

import os
from typing import Union
from abc import ABC, abstractmethod

__all__ = ["PathPos", "PathScore"]

class PathInit(ABC):
    """ Set the directory for results. Leave the project name and record file as not implemented.
     Different sub-Path need to implement the 2 values"""
    def __init__(self):
        self.results_dir: str = 'results'
        self.project_name = self._project_name()
        self.record_file = os.path.join(self.results_dir, self._record_file())

    @abstractmethod
    def _project_name(self):
        raise NotImplementedError

    @abstractmethod
    def _record_file(self):
        raise NotImplementedError


class PathScoreInit(PathInit):
    def __init__(self):
        super().__init__()

    def _project_name(self):
        return 'score'

    def _record_file(self):
        return 'records_700.csv'


class PathPosInit(PathInit):
    def __init__(self):
        super().__init__()

    def _project_name(self):
        return 'pos'

    def _record_file(self):
        return 'records_pos.csv'


class Path(PathInit):
    """
    Common path values are initialized.
    """
    def __init__(self, id: Union[int, str], model_dir: str, check_id_dir: bool=False):
        super().__init__()

        self.slurmlog_dir = os.path.join(self.results_dir, 'slurmlogs')
        self.data_dir = 'dataset'
        self.ori_data_dir = os.path.join(self.data_dir, 'SSc_DeepLearning')
        self.label_excel_fpath = os.path.join(self.data_dir, "GohScores.xlsx")

        if isinstance(id, (int, float)):
            self.id = str(int(id))
        else:
            self.id = id
        self.model_dir = os.path.join(self.results_dir, model_dir)
        self.id_dir = os.path.join(self.model_dir, str(id))  # +'_fold_' + str(args.fold)
        if check_id_dir:  # when infer, do not check
            if os.path.isdir(self.id_dir):  # the dir for this id already exist
                raise Exception('The same id_dir already exists', self.id_dir)

        for dir in [self.slurmlog_dir, self.model_dir, self.data_dir, self.id_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
                print('successfully create directory:', dir)

        self.model_fpath = os.path.join(self.id_dir, 'model.pt')
        self.model_wt_structure_fpath = os.path.join(self.id_dir, 'model_wt_structure.pt')

    def label(self, mode):
        raise NotImplementedError

    def pred(self, mode):
        raise NotImplementedError

    def loss(self, mode):
        raise NotImplementedError

    def data(self, mode):
        raise NotImplementedError


class PathPos(Path, PathPosInit):
    """ Path values for position prediction.

    """
    def __init__(self, id=None, check_id_dir=False) -> None:
        Path.__init__(self, id, 'models_pos', check_id_dir)
        PathPosInit.__init__(self)

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
        """ Dataset directory. Different resample size means different dataset directory."""
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
            raise Exception("wrong resample_z:" + str(resample_z))
        return os.path.join(self.data_dir, res_dir)

    def persis_cache_dir(self):
        return os.path.join(self.results_dir, 'persistent_cache')


class PathScore(Path, PathScoreInit):
    """ Path values for Goh score prediction."""
    def __init__(self, id=None, check_id_dir=False) -> None:
        Path.__init__(self, id, 'models', check_id_dir)
        PathScoreInit.__init__(self)

    def label(self, mode: str):
        return os.path.join(self.id_dir, mode + '_label.csv')

    def pred(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred.csv')

    def pred_int(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_int.csv')

    def pred_end5(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_int_end5.csv')

    def loss(self, mode: str):
        return os.path.join(self.id_dir, mode + '_loss.csv')

    def data(self, mode: str):
        return os.path.join(self.id_dir, mode + '_data.csv')
