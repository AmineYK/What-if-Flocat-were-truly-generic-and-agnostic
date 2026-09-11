#!/bin/bash                                                                                                                                                                                                                                                                                                                                                        
#SBATCH --job-name=flocat-genagn
#SBATCH --output=jobs/%j/%x-%j.out
#SBATCH --error=jobs/%j/%x-%j.err
#SBATCH --partition=gpu
# SBATCH --partition=hpda_mig
# SBATCH --partition=hpda
# SBATCH --nodes=1
#SBATCH --gres=gpu:1
# SBATCH --gres=gpu:a100_3g.40gb
#SBATCH --time 00:30:00
# SBATCH --array=0-13

# environments                                                                                                                                                                                
# ---------------------------------                                                                                                                                                           
module purge
module load aidl/pytorch/2.5.1-cuda12.4
# ---------------------------------          

# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# JOB_DIR="jobs/job_${SLURM_JOB_ID}_${TIMESTAMP}"
# mkdir -p ${JOB_DIR}

echo "Job ID: ${SLURM_JOB_ID}"

export PYTHONUNBUFFERED=1

python embed_images.py --category all --backbone wide_resnet50_2 --out_indices 2 3 --patch_size 3