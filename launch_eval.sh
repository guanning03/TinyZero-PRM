#!/bin/bash
# Launch eval for all 6 checkpoints

mkdir -p logs

sbatch eval.sh countdown_4_30_100_grpo_coeff_0.01
sbatch eval.sh countdown_4_30_100_grpo_coeff_0.0_checkpoint1
sbatch eval.sh countdown_4_30_100_grpo_coeff_0.0_checkpoint2
sbatch eval.sh countdown_4_30_100_grpo_coeff_0.0_checkpoint3
sbatch eval.sh countdown_4_30_100_grpo_coeff_0.1
sbatch eval.sh countdown_4_30_100_grpo_coeff_1.0
