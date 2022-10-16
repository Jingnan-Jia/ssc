import sys
sys.path.append("../../..")
from lung_function.modules.datasets import all_loaders
from lung_function.modules.cam import GradCAM
from tqdm import tqdm
from mlflow import log_metric, log_metrics, log_param, log_params
import mlflow
from mlflow.tracking import MlflowClient
from lung_function.modules.path import PFTPath

from lung_function.scripts.run import Run
from lung_function.modules.set_args import get_args

class Args:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                else:
                    try:
                        value = float(value)  # convert to float value if possible
                        try:
                            if int(value) == value:  # convert to int if possible
                                value = int(value)
                        except Exception:
                            pass
                    except Exception:
                        pass
                setattr(self, key, value)

def main():
    AttentionMethod = "GradCAM"  # or others
    Ex_id = 750
    mlflow.set_tracking_uri("http://nodelogin02:5000")
    experiment = mlflow.set_experiment("lung_fun_db15")
    client = MlflowClient()
    run_ls = client.search_runs(experiment_ids=[experiment.experiment_id],
                                filter_string=f"params.id LIKE '%{Ex_id}%'")
    run_ = run_ls[0]
    args_dt = run_.data.params  # extract the hyper parameters
    args = Args(args_dt)  #convert to object
    args.workers=1

    if AttentionMethod=="GradCAM":
        attention = GradCAM(Ex_id, args_dt, 'last_maxpool')
    else:
        raise Exception(f"Please set the correct AttentionMethod")


    mypath = PFTPath(Ex_id, check_id_dir=False, space=args.ct_sp)

    data_dt = all_loaders(mypath.data_dir, mypath.label_fpath, args, datasetmode='valid', nb=5)
    dataloader = data_dt['valid']

    for data in dataloader:
        batch_pat_id = data['pat_id'].detach().numpy()
        batch_x = data['image']
        batch_y = data['label']
        batch_ori = data['origin'].detach().numpy()
        batch_sp = data['spacing'].detach().numpy()

        for pat_id, image, ori, sp, label in zip(batch_pat_id, batch_x, batch_ori, batch_sp, batch_y):
            attention.run(pat_id, image, ori, sp, label)
            print('Finish pat_id: ', pat_id)
    print("Finish all")


if __name__ == "__main__":
    main()
