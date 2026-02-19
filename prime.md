# PRIME Reward Server 实现总结

本文档总结 `~/exploration2/verl` 中 PRIME reward 计算的架构和实现。

---

## 目录

1. [整体架构](#整体架构)
2. [Reward Manager 注册机制](#reward-manager-注册机制)
3. [PrimeRewardManager 详解](#primerewardmanager-详解)
4. [其他 Reward Manager 对比](#其他-reward-manager-对比)
5. [default_compute_score 路由函数](#default_compute_score-路由函数)
6. [训练循环中的调用流程](#训练循环中的调用流程)
7. [countdown 评分函数差异](#countdown-评分函数差异)
8. [代码位置索引](#代码位置索引)

---

## 整体架构

```
config.reward_model.reward_manager = "prime" | "naive" | "batch" | "dapo" | "multi_thread"
                ↓
    load_reward_manager(config, tokenizer, ...)       # verl/trainer/ppo/reward.py
                ↓
    get_reward_manager_cls("prime")                   # Registry 查找
                ↓
    PrimeRewardManager(tokenizer, compute_score=...) 
                ↓
    trainer.reward_fn = reward_manager                # 赋给 RayPPOTrainer
                ↓
    compute_reward(batch, reward_fn)                  # 训练循环中调用
                ↓
    reward_fn(data, return_dict=True)                 # 即 PrimeRewardManager.__call__
```

### 关键设计：插件式 Reward Manager

exploration2/verl 使用**注册表模式**来管理不同的 Reward Manager。通过配置文件中的 `reward_model.reward_manager` 字段来选择使用哪个实现。

---

## Reward Manager 注册机制

**位置：** `verl/workers/reward_manager/registry.py`

```python
REWARD_MANAGER_REGISTRY = {}

def register(name):
    """装饰器：将 reward manager 类注册到全局注册表中"""
    def decorator(cls):
        REWARD_MANAGER_REGISTRY[name] = cls
        return cls
    return decorator

def get_reward_manager_cls(name):
    """通过名称获取 reward manager 类"""
    return REWARD_MANAGER_REGISTRY[name]
```

已注册的 Reward Manager：

| 名称 | 类 | 文件 |
|------|-----|------|
| `"naive"` | `NaiveRewardManager` | `verl/workers/reward_manager/naive.py` |
| `"prime"` | `PrimeRewardManager` | `verl/workers/reward_manager/prime.py` |
| `"batch"` | `BatchRewardManager` | `verl/workers/reward_manager/batch.py` |
| `"dapo"` | `DAPORewardManager` | `verl/workers/reward_manager/dapo.py` |
| `"multi_thread"` | `MultiThreadNaiveRewardManager` | `verl/workers/reward_manager/multi_thread_naive.py` |

---

## PrimeRewardManager 详解

**位置：** `verl/workers/reward_manager/prime.py`

### 3.1 初始化

```python
@register("prime")
class PrimeRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None,
                 reward_fn_key="data_source", num_processes=32, chunksize=None):
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_processes = num_processes
        self.chunksize = chunksize

        # 创建持久进程池（spawn 上下文，避免 CUDA fork 问题）
        self._mp_context = multiprocessing.get_context("spawn")
        self._pool = self._mp_context.Pool(processes=num_processes)
```

**关键特点：**
- 使用 `multiprocessing.Pool` 进行并行 reward 计算
- 使用 `spawn` context（而非 `fork`），避免 CUDA 相关的 fork 问题
- 进程池在初始化时创建，在整个训练过程中**持久复用**
- 默认 32 个 worker 进程

### 3.2 并行评分 `_run_reward_scoring`

```python
def _run_reward_scoring(self, completions, references, tasks, extra_info=None):
    args_list = [
        (self.compute_score, task, completion, reference, ei)
        for task, completion, reference, ei in zip(tasks, completions, references, extra_info)
    ]
    # 使用 pool.map 并行计算
    chunksize = self.chunksize or max(1, len(args_list) // self.num_processes)
    scores = self._pool.map(_compute_single_score, args_list, chunksize=chunksize)
    return scores
```

**并行策略：**
- 使用 `pool.map` 将任务分配给 worker 进程
- `chunksize` 参数控制每个进程一次接收多少任务（减少 IPC 开销）
- 如果 `pool.map` 失败，会**自动回退**到顺序执行

### 3.3 `verify` 方法

```python
def verify(self, data):
    # batch decode 所有 response（一次性）
    response_ids = data.batch["responses"]
    sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    ground_truth = [item.non_tensor_batch["reward_model"]["ground_truth"] for item in data]
    data_sources = data.non_tensor_batch[self.reward_fn_key]
    extra_info = data.non_tensor_batch.get("extra_info", None)
    
    scores = self._run_reward_scoring(
        completions=sequences_str,
        references=ground_truth,
        tasks=data_sources,
        extra_info=extra_info,
    )
    data.batch["acc"] = torch.tensor(scores, ...)
    return scores
```

**注意：** `verify` 使用 `batch_decode(response_ids, skip_special_tokens=True)`，只解码 **response 部分**（不含 prompt）。

### 3.4 `__call__` 方法

```python
def __call__(self, data: DataProto, return_dict=False):
    if "rm_scores" in data.batch.keys():
        return data.batch["rm_scores"]  # shortcut
    
    reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
    
    # batch decode response
    prompt_length = data.batch["prompts"].shape[-1]
    valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
    sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    scores = self.verify(data)  # 并行计算 scores
    
    for i in range(len(data)):
        reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
    
    if return_dict:
        return {"reward_tensor": reward_tensor}
    return reward_tensor
```

**关键流程：**
1. 检查是否已有 `rm_scores`（shortcut）
2. `batch_decode` 一次性解码所有 response
3. 调用 `verify` → `_run_reward_scoring` 并行计算所有 score
4. 将 score 放在每个 response 的最后一个有效 token 位置
5. 支持 `return_dict=True` 返回字典格式（新版接口）

### 3.5 Worker 函数 `_compute_single_score`

```python
def _compute_single_score(args):
    """在子进程中执行的 worker 函数"""
    evaluation_func, task, completion, reference, extra_info = args
    result = evaluation_func(task, completion, reference, extra_info)
    if isinstance(result, (int, float, bool)):
        return float(result)
    elif isinstance(result, dict):
        return float(result.get("score", 0.0))
    else:
        return float(result[0]) if result else 0.0
```

**注意评分函数的调用签名：** `evaluation_func(task=data_source, completion=response_str, reference=ground_truth, extra_info=extra_info)`

这与 `default_compute_score(data_source, solution_str, ground_truth, extra_info)` 的签名一致。

---

## 其他 Reward Manager 对比

### 4.1 NaiveRewardManager（默认）

**位置：** `verl/workers/reward_manager/naive.py`

- **逐条处理**：for 循环逐个 item 计算 score
- 每个 item 单独 decode（`valid_prompt_ids` + `valid_response_ids`），然后只传 `response_str`
- 直接调用 `self.compute_score(data_source, solution_str=response_str, ground_truth, extra_info)`
- **不使用并行**

### 4.2 BatchRewardManager

**位置：** `verl/workers/reward_manager/batch.py`

- 先 batch decode 所有 response
- 然后**一次性**把所有 `data_sources`, `solution_strs`, `ground_truths`, `extra_infos` 传给 `self.compute_score`
- 适用于 compute_score 本身支持批量输入的场景

### 4.3 DAPORewardManager

**位置：** `verl/workers/reward_manager/dapo.py`

- 逐条处理（类似 NaiveRewardManager）
- 额外支持 **overlong buffer penalty**：如果 response 超长，会根据超长比例扣分

### 4.4 MultiThreadNaiveRewardManager

**位置：** `verl/workers/reward_manager/multi_thread_naive.py`

- 使用 **Ray Actor** 进行并行计算（非 multiprocessing.Pool）
- 每个 Actor 内部有一个 `MathVerifyScorer` 实例
- 支持 **per-item timeout** 和 **per-batch timeout**
- 支持 **majority voting**（多数投票评估）
- 使用 `ray.wait` 进行异步调度

---

## default_compute_score 路由函数

**位置：** `verl/utils/reward_score/__init__.py`

```python
def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, 
                          sandbox_fusion_url=None, concurrent_semaphore=None):
```

路由逻辑：

| data_source | 评分模块 |
|---|---|
| `startswith("maze")` | `maze.judge_maze` |
| `"openai/gsm8k"` | `gsm8k.compute_score` |
| `"lighteval/MATH"`, `"DigitalLearningGmbH/MATH-lighteval"` | `math_verify.compute_score` |
| `"countdown"` | `countdown.compute_score` |
| `"math_dapo"`, `startswith("aime")`, `"amc23"`, `startswith("dapo")` | `math_verify.compute_score` |
| `"numina_*"` | `prime_math.compute_score` |
| `"codecontests"`, `"apps"`, `"codeforces"`, `"taco"` | `sandbox_fusion` 或 `prime_code.compute_score` |
| `"hiyouga/geometry3k"` | `geo3k.compute_score` |
| `"searchR1_*"` | `search_r1_like_qa_em.compute_score` |

**对比 TinyZero-PRM：** TinyZero-PRM 的 `_select_rm_score_fn` 只支持 gsm8k, MATH, multiply, countdown 四种数据源。exploration2/verl 支持更多数据源类型。

---

## 训练循环中的调用流程

**位置：** `verl/trainer/ppo/ray_trainer.py`

```
1. trainer.reward_fn = load_reward_manager(config, tokenizer, ...)
         ↓
2. 训练循环中：
   with marked_timer("reward"):
       if self.use_rm:
           reward_tensor = self.rm_wg.compute_rm_score(batch)  # 模型 RM（可选）
           batch = batch.union(reward_tensor)
       
       if config.reward_model.launch_reward_fn_async:
           # 异步计算 reward（Ray remote task）
           future_reward = compute_reward_async.remote(batch, config, tokenizer)
       else:
           # 同步计算 reward
           reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
         ↓
3. compute_reward(data, reward_fn):
       reward_result = reward_fn(data, return_dict=True)  # 即 PrimeRewardManager.__call__
       reward_tensor = reward_result["reward_tensor"]
       reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
       return reward_tensor, reward_extra_infos_dict
```

### 异步 Reward 计算

exploration2/verl 还支持**异步 reward 计算**：

```python
@ray.remote(num_cpus=1)
def compute_reward_async(data, config, tokenizer):
    """在独立的 Ray worker 中加载 reward manager 并计算 reward"""
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, ...)
    return compute_reward(data, reward_fn)
```

当 `config.reward_model.launch_reward_fn_async=True` 时，reward 计算会和 `old_log_prob`、`ref_log_prob`、`values` 计算**并行执行**。

---

## countdown 评分函数差异

**exploration2/verl** 的 `countdown.compute_score`（`verl/utils/reward_score/countdown.py`）与 **TinyZero-PRM** 存在以下差异：

### 7.1 `extract_solution` — 格式检查更严格

```python
# exploration2/verl 版本
def extract_solution(solution_str):
    for str_to_check in ["<think>", "</think>", "<answer>", "</answer>"]:
        if str_to_check not in solution_str:
            return None  # 必须同时包含所有四个标签
    solution_str = solution_str.split('\n')[-1]
    # ...
```

```python
# TinyZero-PRM 版本
def extract_solution(solution_str):
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None  # 需要 "Assistant:" 或 "<|im_start|>assistant" 前缀
    solution_str = solution_str.split('\n')[-1]
    # ...
```

| 差异点 | exploration2/verl | TinyZero-PRM |
|--------|-------------------|--------------|
| 格式前置检查 | 必须包含 `<think>`, `</think>`, `<answer>`, `</answer>` 四个标签 | 必须包含 `"Assistant:"` 或 `"<|im_start|>assistant"` 前缀 |
| 不含标签时 | `return None`（score=0） | `return None`（score=0） |

### 7.2 `format_score` 默认值不同

| | exploration2/verl | TinyZero-PRM |
|--|---|---|
| `format_score` 默认值 | **0.0** | **0.1** |

exploration2/verl 对格式正确但答案错误的情况不给分（`format_score=0.0`），TinyZero-PRM 给 0.1 分。

### 7.3 输入不同

| | exploration2/verl (PrimeRewardManager) | TinyZero-PRM (RewardManager) |
|--|---|---|
| 传给 `compute_score` 的输入 | `response_str`（仅 response，通过 `batch_decode(response_ids)` 得到） | `sequences_str`（prompt + response，通过 `decode(torch.cat(valid_prompt_ids, valid_response_ids))` 得到） |

---

## 代码位置索引

| 功能 | 文件 | 行数 |
|------|------|------|
| Reward Manager 注册表 | `verl/workers/reward_manager/registry.py` | 1-49 |
| PrimeRewardManager | `verl/workers/reward_manager/prime.py` | 44-181 |
| NaiveRewardManager | `verl/workers/reward_manager/naive.py` | 25-121 |
| BatchRewardManager | `verl/workers/reward_manager/batch.py` | 24-111 |
| DAPORewardManager | `verl/workers/reward_manager/dapo.py` | 25-140 |
| MultiThreadNaiveRewardManager | `verl/workers/reward_manager/multi_thread_naive.py` | 239-715 |
| default_compute_score 路由 | `verl/utils/reward_score/__init__.py` | 19-120 |
| load_reward_manager | `verl/trainer/ppo/reward.py` | 60-108 |
| compute_reward | `verl/trainer/ppo/reward.py` | 111-142 |
| compute_reward_async | `verl/trainer/ppo/reward.py` | 145-152 |
| 训练循环中 reward 调用 | `verl/trainer/ppo/ray_trainer.py` | 1399-1408 |
| countdown.compute_score | `verl/utils/reward_score/countdown.py` | 60-113 |
| prime_math.compute_score | `verl/utils/reward_score/prime_math/__init__.py` | 379-401 |
| prime_code.compute_score | `verl/utils/reward_score/prime_code/__init__.py` | 21-73 |

---

## 总结

1. **插件式架构**：exploration2/verl 使用注册表模式，支持多种 Reward Manager（naive、prime、batch、dapo、multi_thread），通过配置切换。TinyZero-PRM 只有一个硬编码的 `RewardManager` 类。

2. **PRIME 并行策略**：`PrimeRewardManager` 使用 `multiprocessing.Pool`（spawn context，32 进程）并行计算 reward。进程池持久化复用。

3. **Multi-Thread 并行策略**：`MultiThreadNaiveRewardManager` 使用多个 Ray Actor 并行计算，支持 per-item/per-batch timeout、majority voting 等高级功能。

4. **异步 Reward**：训练循环中支持异步 reward 计算（`compute_reward_async`），reward 和 log_prob/ref/values 计算可以并行。

5. **评分函数签名不同**：
   - exploration2/verl：`compute_score(data_source, solution_str, ground_truth, extra_info)`
   - TinyZero-PRM：`compute_score(solution_str, ground_truth, method, format_score, score)`

6. **输入差异**：PrimeRewardManager 只传 response（不含 prompt），TinyZero-PRM 传完整序列（prompt + response）。

7. **countdown 评分差异**：exploration2/verl 要求 `<think>`, `</think>`, `<answer>`, `</answer>` 四个标签都存在；TinyZero-PRM 要求 `"Assistant:"` 或 `"<|im_start|>assistant"` 前缀。`format_score` 默认值也不同（0.0 vs 0.1）。
