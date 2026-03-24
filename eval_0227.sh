#!/bin/bash
#SBATCH --job-name=eval_0227
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log

source /home/jgai/miniconda3/etc/profile.d/conda.sh
conda activate tinyzero_env
export PATH=/home/jgai/miniconda3/envs/tinyzero_env/bin:$PATH
export RAY_TMPDIR=/data/user_data/jgai/tmp/ray
mkdir -p /data/user_data/jgai/tmp/ray

export WANDB_API_KEY=${WANDB_API_KEY:?"Please set WANDB_API_KEY environment variable"}
export HF_TOKEN=${HF_TOKEN:?"Please set HF_TOKEN environment variable"}
export WANDB_ENTITY=Tsinghua-IIIS-AI-Team

CHECKPOINT_NAME=${1:?"Usage: sbatch eval_0227.sh <checkpoint_name>"}

export N_GPUS=4
export BASE_MODEL=/data/user_data/jgai/countdown_20260227/${CHECKPOINT_NAME}/global_step_400
DATA_DIR=/home/jgai/code-guanning/countdown-data/count_down_327680_4_10_50
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=eval_0227_${CHECKPOINT_NAME}
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=640 \
    +data.val_n=128 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.98 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=9999 \
    trainer.test_freq=1 \
    trainer.project_name=TinyZero-Eval \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=2 \
    probe.enable=False \
    reward.use_pool=True \
    reward.num_workers=4 \
    reward.timeout=120
