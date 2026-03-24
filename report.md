# Cluster Report

## 1. SBATCH Headers (1 GPU, max CPUs & MEM)

### General Partition
- QOS: `normal`, 最多 **8 GPU/user**, 最多 10 jobs
- 最大单节点 CPU: **224** (babel-m9-16/20)
- 最大单节点 MEM: **1547000 MB (~1.5TB)** (babel-m9-16/20, babel-s9-16/20 等)

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:1
#SBATCH --mem=1547000M
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log
```

### Preempt Partition
- QOS: `preempt_qos`, 最多 **24 GPU/user**, 最多 24 jobs
- 最大单节点 CPU: **224** (babel-m9-16/20)
- 最大单节点 MEM: **2063000 MB (~2TB)** (babel-u5-16, babel-t5-16, babel-s5-16, H100 节点)

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:1
#SBATCH --mem=2063000M
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log
```

### Debug Partition
- QOS: `debug_qos`, 最多 **2 GPU/user**, 最多 10 jobs
- 最大单节点 CPU: **224** (babel-m9-16/20)
- 最大单节点 MEM: **1547000 MB (~1.5TB)**

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:1
#SBATCH --mem=1547000M
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log
```

> **注意**: 请求最大资源会大幅减少可调度节点数量，排队时间更长。实际使用中建议按需请求（如 32-64 CPUs, 256-512G MEM）。

---

## 2. 当前用户 (jgai) 在 General 和 Preempt 的使用量

**查询时间**: 2026-03-21

### General Partition
| Job ID | Job Name | Status | CPUs | Memory | GPUs | Node |
|--------|----------|--------|------|--------|------|------|
| 6687868 | qwen3-hard04 | RUNNING | 32 | 512G | 4 | babel-x9-16 |
| 6687772 | ppo_verbose | RUNNING | 16 | 256G | 4 | babel-x5-16 |
| 6687872 | qwen3-hard04 | PENDING | 32 | 512G | 4 | (QOSMaxGRESPerUser) |

- **已用 GPU**: 8/8 (已达上限，第3个 job 因此排队)
- **已用 CPU**: 48
- **已用 MEM**: 768G

### Preempt Partition
- **无正在运行的 job**
- 已用 GPU: 0/24

---

## 3. /data/user_data/jgai 磁盘使用情况

| 目录 | 大小 |
|------|------|
| huggingface | 62G |
| cache | 42G |
| TinyZero-Storage | 14G |
| tmp | 750M |
| py_packages | ~0 |
| checkpoints | ~0 |
| **总计** | **118G** |

HuggingFace 仓库 `guanning-ai/countdown_20260225` 总大小约 **81.6 GB**。
