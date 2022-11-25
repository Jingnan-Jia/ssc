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


job_id=$SLURM_JOB_ID
conda activate py38

slurm_dir=results/slurmlogs

##cp script.sh ${slurm_dir}/slurm-${job_id}.shs
scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh
cp mymodules/set_args.py ${slurm_dir}/slurm-${job_id}_set_args.py  # backup setting


# Passing shell variables to ssh
# https://stackoverflow.com/questions/15838927/passing-shell-variables-to-ssh
# The following code will ssh to loginnode and git commit to synchronize commits from different nodes.

# But sleep some time is required otherwise multiple commits by several experiments at the same time
# will lead to commit error: fatal: could not parse HEAD


ssh -tt jjia@nodelogin02 /bin/bash << ENDSSH
echo "Hello, I an in nodelogin02 to do some git operations."
echo $job_id

jobs="$(squeue -u jjia --sort=+i | grep [^0-9]0:[00-60] | awk '{print $1}')"  # "" to make sure multi lines assigned
echo "Total jobs in one minutes:"
echo \$jobs

accu=0
for i in \$jobs; do
    if [[ \$i -eq $job_id ]]; then
    echo start sleep ...
    sleep \$accu
    echo sleep \$accu seconds
    fi

    echo \$i
    ((accu+=5))  # self increament
    echo \$accu
done

cd data/ssc_scoring
echo $job_id
scontrol write batch_script "${job_id}" ssc_scoring/current_script.sh  # for the git commit latter

git add -A
sleep 2  # avoid error: fatal: Could not parse object (https://github.com/Shippable/support/issues/2932)
git commit -m "ssc_scoring, jobid is ${job_id}"
sleep 2
git push origin master
sleep 2
exit
ENDSSH

echo "Hello, I am back in $(hostname) to run the code"


#idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname=$(hostname) --epochs=500 --sys=1 --sampler=0 --pretrained=1 --sys_pro_in_0=0.5 --sys_ratio=0.0 --mode='train' --fold=1 --gen_gg_as_retp=1 --remark="sampler+sys,sys_ratio=0.0, 16 patients in test dataset, including pat_070" &
idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname=$(hostname) --epochs=501 --net='vgg11_HR' --sys=1 --sampler=0 --pretrained=1 --sys_pro_in_0=0.5 --sys_ratio=0.0 --mode='train' --weighted_syn_region=0 --batch_size=10 --fold=1 --gen_gg_as_retp=1 --remark="no pooling, ays_ratio=1, no weight_map (switch x,y), 16 patients in test dataset, interval=5, with pred outpout"




