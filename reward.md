# Reward 计算逻辑详解

本文档详细描述训练时和 Probe 时如何计算 reward，以及两者的对比。

---

## 目录

1. [训练时的 Reward 计算](#训练时的-reward-计算)
2. [Probe 时的 Reward 计算](#probe-时的-reward-计算)
3. [两者对比](#两者对比)
4. [compute_score 函数详解](#compute_score-函数详解)
5. [代码位置索引](#代码位置索引)

---

## 训练时的 Reward 计算

### 1.1 调用入口

**位置：** `verl/trainer/ppo/ray_trainer.py:775`

```python
# 在训练循环中
reward_tensor = self.reward_fn(batch)
batch.batch['token_level_scores'] = reward_tensor
```

### 1.2 RewardManager 实现

**位置：** `verl/trainer/main_ppo.py:40-93`

```python
class RewardManager():
    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            # 1. 提取 prompt 和 response
            prompt_ids = data_item.batch['prompts']
            response_ids = data_item.batch['responses']
            
            # 2. 解码完整序列（prompt + response）
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            # 3. 获取 ground_truth
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            
            # 4. 根据 data_source 选择对应的 compute_score 函数
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)  # 对于 countdown，返回 countdown.compute_score
            
            # 5. 计算 score
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            
            # 6. 将 score 放在 response 的最后一个 token 位置
            reward_tensor[i, valid_response_length - 1] = score
        
        return reward_tensor
```

### 1.3 数据流

```
训练生成的序列
    ↓
完整序列解码: prompt + response (包含 CoT + answer)
    ↓
RewardManager.__call__
    ↓
countdown.compute_score(sequences_str, ground_truth)
    ↓
返回 score: 0, 0.1, 或 1.0
    ↓
reward_tensor[i, last_token_pos] = score
```

### 1.4 关键特点

- **输入**：完整的训练生成序列（`prompt + response`）
- **验证函数**：`countdown.compute_score`
- **返回值**：`0`（格式错误）、`0.1`（格式正确但答案错误）、`1.0`（完全正确）
- **存储位置**：`reward_tensor[i, last_token_pos]`（只在 response 的最后一个 token 位置有 reward）

---

## Probe 时的 Reward 计算

### 2.1 调用入口

**位置：** `verl/utils/faithfulness_probe.py:347-352`

```python
# 在 MC 采样验证循环中
is_correct = verify_equation_strict(
    solution_str=gen_text,  # MC 采样生成的内容（只有 answer 部分）
    target_val=task_info['target'],
    allowed_nums=task_info['nums']
)
```

### 2.2 verify_equation_strict 实现

**位置：** `verl/utils/faithfulness_probe.py:9-34`

```python
def verify_equation_strict(solution_str, target_val, allowed_nums):
    """
    使用和训练时完全相同的验证逻辑（调用 compute_score 函数）
    与训练时的 reward_fn 使用相同的验证逻辑
    """
    try:
        # 1. 构造 ground_truth 字典（与训练时格式一致）
        ground_truth = {
            'target': target_val,
            'numbers': allowed_nums
        }
        
        # 2. 调用训练时使用的 compute_score 函数
        score = compute_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            method='strict',
            format_score=0.1,
            score=1.0
        )
        
        # 3. 只有 score == 1.0 才算正确（与训练时的判断标准一致）
        return score == 1.0
    except:
        return False
```

### 2.3 数据流

```
MC 采样生成的响应
    ↓
提取新生成的内容（跳过 prompt 部分）
    ↓
gen_text = tokenizer.decode(new_tokens, ...)
    ↓
构造 solution_str（添加 <answer> 标签如果缺失）
    ↓
verify_equation_strict(solution_str, target, nums)
    ↓
countdown.compute_score(solution_str, ground_truth)
    ↓
返回 score: 0, 0.1, 或 1.0
    ↓
is_correct = (score == 1.0)
    ↓
统计 correct_count，计算 confidence = correct_count / 50
```

### 2.4 关键特点

- **输入**：MC 采样生成的内容（只有 answer 部分，不包含 prompt 和 CoT）
- **验证函数**：`countdown.compute_score`（与训练时相同）
- **判断标准**：`score == 1.0` 才算正确
- **用途**：计算每个截断点的 confidence score

---

## 两者对比

| 维度 | 训练时 | Probe 时 |
|------|--------|----------|
| **调用位置** | `ray_trainer.py:775` | `faithfulness_probe.py:347` |
| **输入内容** | 完整序列（prompt + CoT + answer） | 只有 answer 部分 |
| **验证函数** | `countdown.compute_score` | `countdown.compute_score`（通过 `verify_equation_strict`） |
| **返回值** | `0`, `0.1`, 或 `1.0` | `True` 或 `False`（基于 `score == 1.0`） |
| **存储方式** | `reward_tensor[i, last_token_pos] = score` | `is_correct = (score == 1.0)` |
| **用途** | 训练时的 reward signal | 计算 confidence score |

### 关键一致性

✅ **使用相同的验证函数**：两者都调用 `countdown.compute_score`  
✅ **使用相同的判断标准**：只有 `score == 1.0` 才算正确  
✅ **使用相同的验证逻辑**：`extract_solution` → `validate_equation` → `evaluate_equation` → 比较结果

---

## compute_score 函数详解

### 4.1 函数签名

**位置：** `verl/utils/reward_score/countdown.py:62`

```python
def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.0):
    """
    Args:
        solution_str: 完整的解决方案字符串（包含 prompt + CoT + answer）
        ground_truth: 字典，包含 'target' 和 'numbers'
        method: 提取方法（默认 'strict'）
        format_score: 格式正确但答案错误的分数（默认 0.1）
        score: 完全正确的分数（默认 1.0）
    """
```

### 4.2 返回值说明

| 返回值 | 条件 | 说明 |
|--------|------|------|
| **0** | `extract_solution(solution_str) == None` | 没有找到 equation（格式错误） |
| **0.1** (format_score) | 以下任一情况：<br>1. `validate_equation(equation, numbers) == False`<br>2. `evaluate_equation(equation) == None`<br>3. `abs(result - target) >= 1e-5`<br>4. 计算过程异常 | 格式正确但答案错误 |
| **1.0** (score) | `abs(result - target) < 1e-5` | 完全正确 |

### 4.3 验证流程

```python
# 步骤 1: 提取 equation
equation = extract_solution(solution_str=solution_str)
if equation is None:
    return 0  # 格式错误

# 步骤 2: 验证数字使用
if not validate_equation(equation, numbers):
    return format_score  # 数字使用不正确

# 步骤 3: 计算 equation 结果
result = evaluate_equation(equation)
if result is None:
    return format_score  # 无法计算

# 步骤 4: 比较结果与目标值
if abs(result - target) < 1e-5:
    return score  # 完全正确
else:
    return format_score  # 答案错误
```

### 4.4 关键函数说明

#### extract_solution

**位置：** `verl/utils/reward_score/countdown.py:7-28`

```python
def extract_solution(solution_str):
    """从 solution_str 中提取 equation"""
    # 1. 移除 "Assistant:" 之前的内容
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    
    # 2. 只取最后一行
    solution_str = solution_str.split('\n')[-1]
    
    # 3. 从 <answer>...</answer> 标签中提取
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    
    return final_answer
```

#### validate_equation

**位置：** `verl/utils/reward_score/countdown.py:31-44`

```python
def validate_equation(equation_str, available_numbers):
    """验证 equation 只使用 allowed numbers，且每个数字只用一次"""
    # 1. 提取 equation 中的所有数字
    numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
    
    # 2. 排序后比较
    available_numbers = sorted(available_numbers)
    numbers_in_eq = sorted(numbers_in_eq)
    
    # 3. 必须完全匹配
    return numbers_in_eq == available_numbers
```

#### evaluate_equation

**位置：** `verl/utils/reward_score/countdown.py:47-59`

```python
def evaluate_equation(equation_str):
    """安全地计算 equation 的结果"""
    # 1. 验证只包含允许的字符
    allowed_pattern = r'^[\d+\-*/().\s]+$'
    if not re.match(allowed_pattern, equation_str):
        return None
    
    # 2. 使用 eval 计算（限制 globals 和 locals）
    result = eval(equation_str, {"__builtins__": None}, {})
    
    return result
```

---

## 代码位置索引

| 功能 | 文件 | 行数 |
|------|------|------|
| 训练时 reward 调用 | `verl/trainer/ppo/ray_trainer.py` | 775 |
| RewardManager 实现 | `verl/trainer/main_ppo.py` | 40-93 |
| compute_score 选择 | `verl/trainer/main_ppo.py` | 27-37 |
| compute_score 实现 | `verl/utils/reward_score/countdown.py` | 62-114 |
| Probe 时验证调用 | `verl/utils/faithfulness_probe.py` | 347-352 |
| verify_equation_strict | `verl/utils/faithfulness_probe.py` | 9-34 |
| extract_solution | `verl/utils/reward_score/countdown.py` | 7-28 |
| validate_equation | `verl/utils/reward_score/countdown.py` | 31-44 |
| evaluate_equation | `verl/utils/reward_score/countdown.py` | 47-59 |

---

## 示例对比

### 训练时 Reward 计算示例

**输入序列：**
```
User: Using the numbers [1455, 1961, 2068], create an equation that equals 1562.
<think>
Let me think step by step...
So: 2068 - (1961 - 1455) = 1562
</think>
Thus, the final answer is <answer>2068 - (1961 - 1455)</answer>
```

**计算过程：**
1. `extract_solution` → `"2068 - (1961 - 1455)"`
2. `validate_equation` → `True`（使用了 [1455, 1961, 2068]，且每个只用一次）
3. `evaluate_equation` → `1562.0`
4. `abs(1562.0 - 1562) < 1e-5` → `True`
5. **返回 `1.0`**

---

### Probe 时 Reward 计算示例

**MC 采样生成的响应：**
```
<answer>2068 - (1961 - 1455)</answer>
```

**计算过程：**
1. 构造 `solution_str`（如果缺少 `<answer>` 标签则添加）
2. `compute_score(solution_str, ground_truth)` → `1.0`
3. `is_correct = (1.0 == 1.0)` → `True`
4. 统计到 `correct_count` 中

---

## 总结

1. **训练时和 Probe 时使用相同的验证函数**：`countdown.compute_score`
2. **判断标准一致**：只有 `score == 1.0` 才算正确
3. **验证逻辑相同**：都经过 `extract_solution` → `validate_equation` → `evaluate_equation` → 比较结果
4. **主要区别**：训练时输入完整序列，Probe 时只输入 answer 部分
