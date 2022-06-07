# SSc score prediction
[![codecov](https://codecov.io/gh/Jingnan-Jia/ssc_scoring/branch/master/graph/badge.svg?token=Z808SDKUFQ)](https://codecov.io/gh/Jingnan-Jia/ssc_scoring)
![example workflow](https://github.com/Jingnan-Jia/ssc_scoring/actions/workflows/test.yml/badge.svg?branch=master)

* `script.sh` is used to submit jobs to slurm
* `set_args.py` stores the super parameters. imported by `run.py`
* `run.py` is the main file to train/infer/continur_train networks for ssc score prediction.
* `records.csv` and `cp_records.csv` is the same, recording/tracking all experiments.
* `confusion.py` is used to get the confusin matrix, accuracy, weighted kappa, MAE, etc. to evaluate trained networks.
------
* `models` directory save the results of each experiments. ID of each experiment is from the `records.csv`.
* `slurmlogs` directory saves the output of training logs.
* `dataset` directory saves the dataset.

## How to run the code?
2 ways:
1. `sbatch script.sh` to submit job to slurm in your server.
2. `run.py --epochs=300 --mode='train' ... ` more arguments can be found in `set_args.py`.

### Predict Goh scores from 2d CT slices
#### train and inference
#### inference

### Predict 5 positions from 3d CT scans
#### train and inference
#### inference