#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
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

##cp script.sh ${slurm_dir}/slurm-${job_id}.shs
scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh
cp mymodules/set_args.py ${slurm_dir}/slurm-${job_id}_set_args.py  # backup setting

#idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname=$(hostname) --epochs=500 --sys=1 --sampler=0 --pretrained=1 --sys_pro_in_0=0.5 --sys_ratio=0.0 --mode='train' --fold=1 --gen_gg_as_retp=1 --remark="sampler+sys,sys_ratio=0.0, 16 patients in test dataset, including pat_070" &
idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname=$(hostname) --epochs=501 --net='convnext_tiny' --sys=1 --sampler=0 --pretrained=1 --sys_pro_in_0=0.0 --sys_ratio=0.5 --mode='train' --batch_size=10 --fold=1 --gen_gg_as_retp=1 --remark="16 patients in test dataset, interval=25, with pred outpout"




