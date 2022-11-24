#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
##SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com

eval "$(conda shell.bash hook)"

conda activate py38

job_id=$SLURM_JOB_ID
slurm_dir=results/slurmlogs

#cp script.sh ${slurm_dir}/slurm-${job_id}.sh
scontrol write batch_script ${job_id} ${slurm_dir}/slurm-${job_id}_args.sh
#cp mymodules/set_args_pos.py ${slurm_dir}/slurm-${job_id}_set_args.py  # backup setting


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
git commit -m "ssc_scoring_pos, jobid is ${job_id}"
sleep 2
git push origin master
sleep 2
exit
ENDSSH

echo "Hello, I am back in $(hostname) to run the code"


idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run_pos.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --net="x3d_xs" --train_on_level=0 --batch_size=1 --mode='train' --infer_2nd=0 --eval_id=0 --level_node=0 --fold=1 --remark="vgg11, train/testing dataset is the same as PFT"
#idx=1; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run_pos.py 2>${slurm_dir}/slurm-${job_id}_$idx.err 1>${slurm_dir}/slurm-${job_id}_$idx.out --outfile=${slurm_dir}/slurm-${job_id}_$idx --hostname="$(hostname)" --net="vgg11_3d" --train_on_level=2 --mode='infer' --infer_2nd=0 --eval_id=465 --level_node=0 --fold=4 --remark="infer fine net" &

#wait