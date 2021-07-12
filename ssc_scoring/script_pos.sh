#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=6
##SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com

eval "$(conda shell.bash hook)"

conda activate py37

job_id=$SLURM_JOB_ID
slurm_dir=results/slurmlogs

#cp script.sh ${slurm_dir}/slurm-${job_id}.sh
scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh
cp mymodules/set_args_pos.py ${slurm_dir}/slurm-${job_id}_set_args.py  # backup setting

idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run_pos.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --net="vgg11_3d" --train_on_level=2 --mode='train' --infer_2nd=0 --eval_id=0 --level_node=0 --fold=3 --remark="train fine net" &
idx=1; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run_pos.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --net="vgg11_3d" --train_on_level=2 --mode='train' --infer_2nd=0 --eval_id=0 --level_node=0 --fold=4 --remark="train fine net" &

wait




