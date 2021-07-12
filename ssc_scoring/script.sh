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

##cp script.sh ${slurm_dir}/slurm-${job_id}.sh
scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh
cp mymodules/set_args.py ${slurm_dir}/slurm-${job_id}_set_args.py  # backup setting

idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>slurmlogs/slurm-${job_id}_$idx.err 1>slurmlogs/slurm-${job_id}_$idx.out --outfile=slurmlogs/slurm-${job_id}_$idx --hostname=$(hostname) --epochs=1000 --net='vgg11_bn' --sys=0 --sampler=0 --pretrained=0 --sys_pro_in_0=0.5 --sys_ratio=0.0 --mode='train' --fold=3 --remark="vgg11, sys, batchsize=10, ellipselenth=100+200, nosampler, gen_gg_as_retp correctlys" &
idx=1; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>slurmlogs/slurm-${job_id}_$idx.err 1>slurmlogs/slurm-${job_id}_$idx.out --outfile=slurmlogs/slurm-${job_id}_$idx --hostname=$(hostname) --epochs=1000 --net='vgg11_bn' --sys=0 --sampler=0 --pretrained=0 --sys_pro_in_0=0.5 --sys_ratio=0.0 --mode='train' --fold=4 --remark="vgg11, sys, batchsize=10, ellipselenth=100+200, nosampler, gen_gg_as_retp correctlys" &


wait




