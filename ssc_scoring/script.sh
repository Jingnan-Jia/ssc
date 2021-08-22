#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=6
##SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH -e results/slurmlogs/slurm-%j.err
#SBATCH -o results/slurmlogs/slurm-%j.out
##SBATCH --output=output_vessel_only_medium.txt
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com

eval "$(conda shell.bash hook)"

conda activate py38

job_id=$SLURM_JOB_ID
slurm_dir=results/slurmlogs

##cp script.sh ${slurm_dir}/slurm-${job_id}.sh
scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh
cp mymodules/set_args.py ${slurm_dir}/slurm-${job_id}_set_args.py  # backup setting

idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname=$(hostname) --epochs=1000 --net='resnet18' --sys=1 --sampler=0 --pretrained=1 --sys_pro_in_0=0.5 --sys_ratio=0.0 --mode='train' --fold=3 --gen_gg_as_retp=1 --remark="use multiple seeds, from -1500 to 1500 for img and sys seed, gg_as_ret, " &
idx=1; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname=$(hostname) --epochs=1000 --net='resnet18' --sys=1 --sampler=0 --pretrained=1 --sys_pro_in_0=0.5 --sys_ratio=0.0 --mode='train' --fold=4 --gen_gg_as_retp=1 --remark="use multiple seeds, from -1500 to 1500 for img and sys seed, gg_as_ret, " &


wait




