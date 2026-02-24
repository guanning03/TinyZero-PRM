#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=production
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log

# ── Activate conda environment ──
source /home/azanette/miniconda3/etc/profile.d/conda.sh
conda activate tinyzero

export WANDB_API_KEY=256879fdda25bc1fb8ee4f0310e71615e92f75c9
export HF_TOKEN=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
export WANDB_ENTITY=Tsinghua-IIIS-AI-Team

export N_GPUS=8
export OVERCONF_COEFF=${2:-0.0}
export BASE_MODEL=/home/azanette/TinyZero-PRM/checkpoints/base_models/Qwen2.5-3B
export BASE_DATA_DIR=${1:-count_down_327680_3_3700_7400}
DATA_DIR=countdown-data/${BASE_DATA_DIR}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=3b_grpo2_${BASE_DATA_DIR}_coeff_${OVERCONF_COEFF}
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=640 \
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
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=400 \
    trainer.test_freq=10 \
    trainer.project_name=TinyZero-PRM-Design1 \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=9999 \
    trainer.total_training_steps=402 \
    probe.overconf_coeff=$OVERCONF_COEFF \
    probe.enable=True \
    probe.num_truncations=5 \
    probe.mc_samples=10 \
    probe.mc_max_tokens=32 \
    probe.num_splits=1 \
    reward.use_pool=True \
    reward.num_workers=4 \
    reward.timeout=120 
