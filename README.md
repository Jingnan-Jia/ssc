# SSc score prediction

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
`run.py`

### Predict 5 positions from 3d CT scans
`run_pos.py`
## Where is the trained models? 
Because github does not allow large file repository, so I put it at [google drive](https://drive.google.com/drive/folders/1ZqxFKgQNO5t1Ccb6CqIXSpM6hEzkffQF?usp=drive_link).
