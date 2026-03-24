#!/bin/bash
#SBATCH --job-name=download_model
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/download-%j.log
#SBATCH --error=logs/download-%j.log

source /home/jgai/miniconda3/etc/profile.d/conda.sh
conda activate tinyzero_env
export PATH=/home/jgai/miniconda3/envs/tinyzero_env/bin:$PATH

export HF_TOKEN=${HF_TOKEN:?"Please set HF_TOKEN environment variable"}

TARGET_DIR=/data/user_data/jgai/countdown_20260227
mkdir -p $TARGET_DIR

echo "Downloading guanning-ai/countdown_20260227 to $TARGET_DIR"
huggingface-cli download guanning-ai/countdown_20260227 --local-dir $TARGET_DIR

echo "Download complete. Contents:"
ls -la $TARGET_DIR
