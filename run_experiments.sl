#!/bin/bash                                                                                                                                                                                                                                                                                                                                                        
#SBATCH --job-name=flocat-genagn
#SBATCH --output=jobs/%j/%x-%j.out
#SBATCH --error=jobs/%j/%x-%j.err
# SBATCH --partition=gpu
# SBATCH --partition=hpda_mig
# SBATCH --partition=gpu_h200
#SBATCH --partition=hpda
# SBATCH --nodes=1
# SBATCH --gres=gpu:1
# SBATCH --gres=gpu:a100_3g.40gb
#SBATCH --time 00:30:00
#SBATCH --array=0-14

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

topics=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush" "transistor" "wood" "zipper")

topic=${topics[$SLURM_ARRAY_TASK_ID]}


# python run_flocat.py  --data_root /home/2017025/ayouce01/FLOCAT-GENAGN/What-if-Flocat-were-truly-generic-and-agnostic/data/embeddings \
#                       --embedding_type vit_base_patch16_224 --inlier_topic "$topic" --dataset_name MVTecAD --output_file Results/vit_base_patch16_224/mv_tec_ad/results.txt

# python run_flocat.py  --data_root /home/2017025/ayouce01/FLOCAT-GENAGN/What-if-Flocat-were-truly-generic-and-agnostic/data/embeddings \
# --embedding_type wide_resnet50_2 --inlier_topic "pill" --dataset_name MVTecAD --output_file Results/wide_resnet50_2/mv_tec_ad/results.txt


python run_flocat.py  --data_root /home/2017025/ayouce01/FLOCAT-GENAGN/What-if-Flocat-were-truly-generic-and-agnostic/data/embeddings/diypatch_level \
                      --embedding_type vit_base_patch16_224 --inlier_topic "$topic" --dataset_name MVTecAD --output_file Results/mv_tec_ad/diypatch_level/vit_base_patch16_224/results.txt \
                      --diy_patch_size 64

# python embed_images.py --category all --backbone wide_resnet50_2 --out_indices 2 3 --patch_size 3


# python embed_images.py --model_type vit --vit_model_name vit_base_patch16_224 --image_size 256 --category "$topic" --output "data/embeddings/image_level" --is_cls
# python embed_images.py --model_type vit --vit_model_name vit_base_patch16_224 --image_size 256 --category "$topic" --output "data/embeddings/vitpatch_level"
# python embed_images.py --model_type vit_manual_patches --manual_patch_size 32 --image_size 256 --vit_input_size 256 --category "$topic" --output "data/embeddings/diypatch_level"
# python embed_images.py --model_type vit_manual_patches --manual_patch_size 64 --image_size 256 --vit_input_size 256 --category "bottle" --output "data/embeddings/diypatch_level"