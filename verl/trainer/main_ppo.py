# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import multiprocessing
import time

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    else:
        raise NotImplementedError


def _pool_compute_single_score(args):
    """Worker function for multiprocessing Pool (runs in a spawned process).

    Each worker re-imports the scoring modules to avoid pickling issues with
    the ``spawn`` context.  The function is deliberately kept at module level
    so that it is pickle-able across processes.
    """
    data_source, solution_str, ground_truth = args
    try:
        # Re-import inside the worker – necessary for 'spawn' context
        from verl.utils.reward_score import countdown as _countdown
        from verl.utils.reward_score import gsm8k as _gsm8k
        from verl.utils.reward_score import math as _math
        from verl.utils.reward_score import multiply as _multiply

        if data_source == 'openai/gsm8k':
            fn = _gsm8k.compute_score
        elif data_source == 'lighteval/MATH':
            fn = _math.compute_score
        elif "multiply" in data_source or "arithmetic" in data_source:
            fn = _multiply.compute_score
        elif "countdown" in data_source:
            fn = _countdown.compute_score
        else:
            return 0.0

        score = fn(solution_str=solution_str, ground_truth=ground_truth)
        return float(score) if isinstance(score, (int, float, bool)) else 0.0
    except Exception as e:
        print(f"[PoolReward] Error computing score: {e}")
        return 0.0


def _pool_compute_probe_score(args):
    """Worker function for probe scoring in a spawned process.

    Same as ``_pool_compute_single_score`` but passes ``format_score=0`` so
    that only truly-correct answers count (no partial credit for format).
    """
    data_source, solution_str, ground_truth = args
    try:
        from verl.utils.reward_score import countdown as _countdown
        from verl.utils.reward_score import gsm8k as _gsm8k
        from verl.utils.reward_score import math as _math
        from verl.utils.reward_score import multiply as _multiply

        if data_source == 'openai/gsm8k':
            fn = _gsm8k.compute_score
        elif data_source == 'lighteval/MATH':
            fn = _math.compute_score
        elif "multiply" in data_source or "arithmetic" in data_source:
            fn = _multiply.compute_score
        elif "countdown" in data_source:
            fn = _countdown.compute_score
        else:
            return 0.0

        score = fn(solution_str=solution_str, ground_truth=ground_truth, format_score=0)
        return float(score) if isinstance(score, (int, float, bool)) else 0.0
    except Exception as e:
        print(f"[PoolProbe] Error computing score: {e}")
        return 0.0


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


def create_shared_scoring_pool(num_workers=4):
    """Create a single shared multiprocessing Pool (spawn context, CUDA-safe).

    This pool should be created ONCE and passed to all PoolRewardManager
    instances **and** to the RayPPOTrainer for probe scoring.  This avoids
    the previous bug where 3 separate pools × 16 workers = 48 extra
    processes that caused OOM.
    """
    print(f"[SharedPool] Creating ONE Pool with {num_workers} workers (spawn context)...")
    t0 = time.time()
    pool = multiprocessing.get_context("spawn").Pool(processes=num_workers)
    print(f"[SharedPool] Pool ready in {time.time() - t0:.2f}s  "
          f"(shared across train-reward, val-reward, and probe-reward)")
    return pool


class PoolRewardManager():
    """Reward manager backed by a shared ``multiprocessing.Pool``.

    Advantages over the sequential :class:`RewardManager`:

    * **Parallel scoring** – all items in a batch are scored concurrently
      across pool workers, reducing wall-clock time.
    * **Batch-level timeout** – if the pool does not finish within
      ``timeout`` seconds the call falls back to sequential execution so
      that training is never blocked indefinitely.
    """

    def __init__(self, tokenizer, num_examine, pool, num_workers=4, timeout=120) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.num_workers = num_workers
        self.timeout = timeout  # per-batch timeout in seconds
        self._pool = pool       # shared pool – NOT owned by this instance
        print(f"[PoolReward] Attached to shared pool (workers={num_workers}, timeout={timeout}s)")

    # Pool is shared — do NOT close it in __del__

    def __call__(self, data: DataProto):
        batch_size = len(data)
        print(f"[PoolReward] ── Start reward computation (batch_size={batch_size}) ──")

        if 'rm_scores' in data.batch.keys():
            print(f"[PoolReward] Found pre-computed rm_scores, skipping scoring")
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # ── Step 1: decode every sample in the main process (fast, CPU) ──
        print(f"[PoolReward] Step 1/3: Decoding {batch_size} samples...")
        t_start = time.time()
        items = []  # (index, data_source, seq_str, ground_truth, valid_resp_len)
        data_source_counts = {}
        for i in range(batch_size):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            items.append((i, data_source, sequences_str, ground_truth,
                          valid_response_length.item()))
            data_source_counts[data_source] = data_source_counts.get(data_source, 0) + 1
        t_decode = time.time() - t_start
        ds_summary = ", ".join(f"{k}={v}" for k, v in data_source_counts.items())
        print(f"[PoolReward] Step 1/3 done: decoded {batch_size} samples in {t_decode:.3f}s  "
              f"(data_sources: {ds_summary})")

        # ── Step 2: parallel score computation via Pool ──
        args_list = [(ds, seq, gt) for (_, ds, seq, gt, _) in items]
        chunksize = max(1, len(args_list) // self.num_workers)
        print(f"[PoolReward] Step 2/3: Submitting {len(args_list)} tasks to pool "
              f"(workers={self.num_workers}, chunksize={chunksize}, timeout={self.timeout}s)...")
        t_score_start = time.time()

        # IMPORTANT: When this file runs as __main__ (python -m verl.trainer.main_ppo),
        # functions defined here are under __main__ module, which spawn workers can't find.
        # We must import the function via its package path so pickle uses the correct module name.
        from verl.trainer.main_ppo import _pool_compute_single_score as _pool_fn

        scoring_method = 'pool'
        try:
            async_result = self._pool.map_async(
                _pool_fn, args_list, chunksize=chunksize)
            scores = async_result.get(timeout=self.timeout)
        except multiprocessing.TimeoutError:
            scoring_method = 'sequential(timeout-fallback)'
            print(f"[PoolReward] ⚠ WARNING: Pool timed out after {self.timeout}s! "
                  f"Falling back to sequential scoring...")
            scores = [_pool_fn(a) for a in args_list]
        except Exception as e:
            scoring_method = f'sequential(error-fallback: {e})'
            print(f"[PoolReward] ⚠ WARNING: Pool.map failed ({e})! "
                  f"Falling back to sequential scoring...")
            scores = [_pool_fn(a) for a in args_list]
        t_score = time.time() - t_score_start
        print(f"[PoolReward] Step 2/3 done: scored {len(scores)} items in {t_score:.3f}s  "
              f"(method={scoring_method})")

        # ── Step 3: fill reward tensor & compute stats ──
        for idx, (i, _, _, _, valid_resp_len) in enumerate(items):
            reward_tensor[i, valid_resp_len - 1] = scores[idx]

        # Compute detailed statistics
        n_total = len(scores)
        n_correct = sum(1 for s in scores if s >= 1.0)
        n_format = sum(1 for s in scores if 0 < s < 1.0)
        n_zero = sum(1 for s in scores if s == 0.0)
        mean_score = sum(scores) / n_total if n_total else 0.0

        print(f"[PoolReward] Step 3/3: Reward tensor filled. Stats: "
              f"mean={mean_score:.4f}, correct={n_correct}/{n_total} ({100*n_correct/n_total:.1f}%), "
              f"format_only={n_format}, zero={n_zero}")
        print(f"[PoolReward] ── Done: {n_total} scores in {t_decode + t_score:.3f}s total "
              f"(decode={t_decode:.3f}s + score={t_score:.3f}s) ──")

        # Print a few example sequences for debugging
        already_printed = {}
        for idx, (i, ds, seq_str, gt, vrl) in enumerate(items):
            if ds not in already_printed:
                already_printed[ds] = 0
            if already_printed[ds] < self.num_examine:
                already_printed[ds] += 1
                print(f"[PoolReward] Example [{ds}] score={scores[idx]:.2f}: {seq_str[:200]}...")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # Choose reward manager: PoolRewardManager (parallel + timeout) or sequential
    reward_cfg = config.get('reward', {})
    use_pool = reward_cfg.get('use_pool', False)
    shared_pool = None  # will be set if use_pool=True

    if use_pool:
        num_workers = reward_cfg.get('num_workers', 4)
        timeout = reward_cfg.get('timeout', 120)
        # Create ONE shared pool for ALL scoring (train reward, val reward, probe reward)
        shared_pool = create_shared_scoring_pool(num_workers=num_workers)
        print(f"[main] Using PoolRewardManager (shared pool, workers={num_workers}, timeout={timeout}s)")
        reward_fn = PoolRewardManager(tokenizer=tokenizer, num_examine=10,
                                      pool=shared_pool, num_workers=num_workers, timeout=timeout)
        val_reward_fn = PoolRewardManager(tokenizer=tokenizer, num_examine=10,
                                          pool=shared_pool, num_workers=num_workers, timeout=timeout)
    else:
        print("[main] Using sequential RewardManager")
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=10)
    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=10)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            shared_pool=shared_pool)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
