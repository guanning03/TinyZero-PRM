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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.model import compute_position_id_with_mask

WorkerType = Type[Worker]


# ── Probe helpers (module-level for testability) ──────────────────────────────

def find_token_seq(tokens: 'torch.Tensor', pattern: 'torch.Tensor') -> int:
    """Find first occurrence of *pattern* in *tokens*.

    Both arguments are 1-D ``torch.Tensor`` of token IDs.
    Returns the starting index, or ``-1`` if not found.
    """
    import torch
    plen = len(pattern)
    for pos in range(len(tokens) - plen + 1):
        if torch.equal(tokens[pos:pos + plen], pattern):
            return pos
    return -1


def find_cot_boundaries(
    valid_response: 'torch.Tensor',
    think_open_ids: 'torch.Tensor',
    think_close_ids: 'torch.Tensor',
    answer_open_ids: 'torch.Tensor',
):
    """Detect CoT start / end inside a response token sequence.

    Returns ``(cot_start, cot_end, boundary_tag)`` where:

    * **cot_start** – index of the first CoT token (right after ``<think>``
      if present, otherwise 0).
    * **cot_end** – index one-past the last CoT token.  Determined by the
      first boundary marker found in priority order:

      1. ``</think>`` → standard case.
      2. ``<answer>`` → fallback when the prompt already ends with ``<think>``
         and the model never generates ``</think>``.
      3. full response length → last resort.

    * **boundary_tag** – human-readable label indicating which boundary was
      used (useful for logging / debugging).
    """
    vrl = len(valid_response)

    # ── cot_start ──
    think_s = find_token_seq(valid_response, think_open_ids)
    if think_s >= 0:
        cot_start = think_s + len(think_open_ids)
    else:
        cot_start = 0

    # ── cot_end ──
    think_e = find_token_seq(valid_response, think_close_ids)
    if think_e >= 0:
        cot_end = think_e
        boundary_tag = '</think>'
    else:
        answer_s = find_token_seq(valid_response, answer_open_ids)
        if answer_s >= 0:
            cot_end = answer_s
            boundary_tag = '<answer>(fallback)'
        else:
            cot_end = vrl
            boundary_tag = 'none(full response)'

    return cot_start, cot_end, boundary_tag


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 shared_pool=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # Use the shared scoring pool passed from main_task (if any)
        reward_cfg = config.get('reward', {})
        if shared_pool is not None:
            self._score_pool = shared_pool
            self._score_pool_timeout = reward_cfg.get('timeout', 120)
            print(f"[RayPPOTrainer] Using shared scoring pool for probe (timeout={self._score_pool_timeout}s)")
        else:
            self._score_pool = None
            self._score_pool_timeout = 120

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _run_probe_single_batch(self, sub_batch: DataProto, batch_start_idx: int, 
                                 probe_cfg, tokenizer, pad_token_id, suffix_ids, 
                                 think_open_ids, think_close_ids, answer_open_ids,
                                 timing_raw) -> torch.Tensor:
        """Run probe on a single sub-batch. Internal helper for _run_probe.

        Args:
            sub_batch: a subset of the full batch
            batch_start_idx: the starting index of this sub-batch in the original batch
            probe_cfg: probe configuration
            tokenizer: tokenizer instance
            pad_token_id: padding token ID
            suffix_ids: encoded suffix tokens
            think_open_ids: encoded <think> tokens
            think_close_ids: encoded </think> tokens
            answer_open_ids: encoded <answer> tokens (fallback CoT boundary)
            timing_raw: dict to record probe_gen and probe_reward timings

        Returns:
            probe_scores: Tensor of shape (sub_batch_size, num_truncations + 1)
        """
        num_truncations = probe_cfg.num_truncations
        mc_samples = probe_cfg.mc_samples
        mc_max_tokens = probe_cfg.mc_max_tokens
        suffix_str = probe_cfg.suffix

        sub_batch_size = len(sub_batch)
        num_trunc_points = num_truncations + 1

        # ── 1. Extract valid prompt & response tokens for each sample ──
        prompt_all = sub_batch.batch['prompts']
        response_all = sub_batch.batch['responses']
        attention_mask = sub_batch.batch['attention_mask']
        prompt_max_len = prompt_all.shape[1]

        prompt_mask = attention_mask[:, :prompt_max_len]
        response_mask = attention_mask[:, prompt_max_len:]
        valid_prompt_lengths = prompt_mask.sum(dim=1).long()
        valid_response_lengths = response_mask.sum(dim=1).long()

        # ── 2. Find CoT boundaries & build probe prompts ──
        all_probe_token_ids = []

        for i in range(sub_batch_size):
            vpl = valid_prompt_lengths[i].item()
            vrl = valid_response_lengths[i].item()

            valid_prompt = prompt_all[i, prompt_max_len - vpl:]
            valid_response = response_all[i, :vrl]

            cot_start, cot_end, boundary_tag = find_cot_boundaries(
                valid_response, think_open_ids, think_close_ids, answer_open_ids)

            cot_len = cot_end - cot_start

            if batch_start_idx + i == 0:
                cot_text_preview = tokenizer.decode(
                    valid_response[cot_start:min(cot_start + 30, cot_end)],
                    skip_special_tokens=False)
                print(f'[Probe] sample 0: cot_start={cot_start}, cot_end={cot_end}, '
                      f'cot_len={cot_len}, response_len={vrl}, boundary={boundary_tag}')
                print(f'  CoT preview: {cot_text_preview!r}...')

            for k in range(num_truncations + 1):
                cot_trunc = round(cot_len * k / num_truncations)
                trunc_end = cot_start + cot_trunc
                truncated_response = valid_response[:trunc_end]
                probe_tokens = torch.cat([valid_prompt, truncated_response, suffix_ids], dim=0)
                all_probe_token_ids.append(probe_tokens)

        # ── 3. Pad to uniform length (left-pad) and build attention_mask, position_ids ──
        num_probes = len(all_probe_token_ids)
        max_probe_len = max(t.shape[0] for t in all_probe_token_ids)

        padded_input_ids = torch.full((num_probes, max_probe_len), pad_token_id, dtype=torch.long)
        probe_attention_mask = torch.zeros((num_probes, max_probe_len), dtype=torch.long)

        for j, tokens in enumerate(all_probe_token_ids):
            length = tokens.shape[0]
            padded_input_ids[j, max_probe_len - length:] = tokens
            probe_attention_mask[j, max_probe_len - length:] = 1

        probe_position_ids = compute_position_id_with_mask(probe_attention_mask)

        # ── 4. Package into DataProto and send to vLLM ──
        from tensordict import TensorDict
        probe_batch = TensorDict({
            'input_ids': padded_input_ids,
            'attention_mask': probe_attention_mask,
            'position_ids': probe_position_ids,
        }, batch_size=num_probes)

        probe_data = DataProto(batch=probe_batch)
        probe_data.meta_info = {
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': pad_token_id,
            'recompute_log_prob': False,
            'do_sample': True,
            'probe_n': mc_samples,
            'probe_max_tokens': mc_max_tokens,
        }

        # ── probe_gen: vLLM generation ──
        with _timer('probe_gen', timing_raw):
            probe_data_padded, probe_pad_size = pad_dataproto_to_divisor(probe_data, self.actor_rollout_wg.world_size)
            print(f'[Probe] Generating {num_probes} probe prompts (padded to {len(probe_data_padded)}) '
                  f'× {mc_samples} MC samples for sub-batch (samples {batch_start_idx}-{batch_start_idx + sub_batch_size - 1})')
            probe_output_padded = self.actor_rollout_wg.generate_probe_sequences(probe_data_padded)
            probe_output = unpad_dataproto(probe_output_padded, pad_size=probe_pad_size)
            print(f'[Probe] Generation complete for sub-batch (samples {batch_start_idx}-{batch_start_idx + sub_batch_size - 1})')
            
            # Release GPU memory after generation
            del probe_data_padded, probe_output_padded, probe_data
            torch.cuda.empty_cache()

        # ── probe_reward: scoring ──
        with _timer('probe_reward', timing_raw):
            print(f'[Probe] Computing rewards for sub-batch (samples {batch_start_idx}-{batch_start_idx + sub_batch_size - 1})...')
            probe_responses = probe_output.batch['probe_responses']
            probe_scores = torch.zeros(sub_batch_size, num_trunc_points, dtype=torch.float32)

            # --- Step A: Decode all probe responses in the main process (fast, CPU) ---
            import time as _time
            t_decode_start = _time.time()

            # Pre-collect per-sample metadata
            sample_data_sources = []
            sample_ground_truths = []
            for i in range(sub_batch_size):
                sample_data_sources.append(sub_batch.non_tensor_batch['data_source'][i])
                sample_ground_truths.append(sub_batch.non_tensor_batch['reward_model'][i]['ground_truth'])

            # Build flat list of scoring tasks: one per (sample, trunc, mc)
            total_tasks = sub_batch_size * num_trunc_points * mc_samples
            scoring_args = []          # (data_source, solution_str, ground_truth)
            debug_first_gen = {}       # (i, k) -> gen_str for first MC sample (debug printing)

            for i in range(sub_batch_size):
                data_source = sample_data_sources[i]
                ground_truth = sample_ground_truths[i]
                orig_idx = batch_start_idx + i

                if orig_idx == 0:
                    print(f'[Probe debug] sample 0: ground_truth={ground_truth}')

                for k in range(num_trunc_points):
                    probe_idx = i * num_trunc_points + k
                    start_row = probe_idx * mc_samples

                    for m_offset in range(mc_samples):
                        gen_ids = probe_responses[start_row + m_offset]
                        gen_mask = (gen_ids != pad_token_id)
                        valid_gen_ids = gen_ids[gen_mask]
                        gen_str = tokenizer.decode(valid_gen_ids, skip_special_tokens=False)

                        clean_solution = f"<|im_start|>assistant\n{suffix_str}{gen_str}"
                        scoring_args.append((data_source, clean_solution, ground_truth))

                        if m_offset == 0:
                            debug_first_gen[(i, k)] = gen_str

            t_decode = _time.time() - t_decode_start

            # --- Step B: Score all tasks (parallel via pool, or sequential fallback) ---
            t_score_start = _time.time()

            if self._score_pool is not None:
                from verl.trainer.main_ppo import _pool_compute_probe_score
                chunksize = max(1, total_tasks // 64)  # reasonable chunk size
                try:
                    async_result = self._score_pool.map_async(
                        _pool_compute_probe_score, scoring_args, chunksize=chunksize)
                    all_scores = async_result.get(timeout=self._score_pool_timeout)
                    scoring_method = 'pool'
                except Exception as e:
                    print(f'[Probe] WARNING: Pool scoring failed ({e}), falling back to sequential')
                    all_scores = [_pool_compute_probe_score(a) for a in scoring_args]
                    scoring_method = 'sequential(fallback)'
            else:
                from verl.trainer.main_ppo import _select_rm_score_fn
                all_scores = []
                for ds, sol, gt in scoring_args:
                    fn = _select_rm_score_fn(ds)
                    s = fn(solution_str=sol, ground_truth=gt, format_score=0)
                    all_scores.append(float(s) if isinstance(s, (int, float, bool)) else 0.0)
                scoring_method = 'sequential'

            t_score = _time.time() - t_score_start

            # --- Step C: Aggregate scores into probe_scores tensor ---
            score_idx = 0
            for i in range(sub_batch_size):
                orig_idx = batch_start_idx + i
                for k in range(num_trunc_points):
                    num_correct = 0
                    first_score = None
                    for m_offset in range(mc_samples):
                        s = all_scores[score_idx]
                        if m_offset == 0:
                            first_score = s
                        if s >= 1.0:
                            num_correct += 1
                        score_idx += 1

                    probe_scores[i, k] = num_correct / mc_samples

                    if orig_idx == 0:
                        frac = k / num_truncations
                        first_gen_str = debug_first_gen.get((i, k), '')
                        print(f'  [Probe sample 0] trunc={frac:.1f} | gen: {first_gen_str!r} | score={first_score} | pass_rate={num_correct}/{mc_samples}')

            # Release GPU memory after scoring
            del probe_output, probe_responses
            torch.cuda.empty_cache()
            print(f'[Probe] Reward scoring complete for sub-batch (samples {batch_start_idx}-{batch_start_idx + sub_batch_size - 1}): '
                  f'{total_tasks} tasks, method={scoring_method}, decode={t_decode:.3f}s, score={t_score:.3f}s')

        return probe_scores

    def _run_probe(self, batch: DataProto, timing_raw: Dict[str, float]) -> torch.Tensor:
        """Run faithfulness probe:
        1. Extract CoT from each response. The CoT boundary is determined by:
           - Primary: tokens between <think> and </think>
           - Fallback: tokens before <answer> (when </think> is not in the
             response, e.g. because the prompt already ends with <think>)
        2. Truncate only the CoT at fractions 0/N, 1/N, ..., N/N
        3. Construct: [prompt] + [response up to boundary] + [partial CoT] + [suffix]
           where suffix = "</think> Thus, the final answer is <answer> "
        4. Generate MC completions and score them
        
        Supports splitting the batch into K sub-batches to reduce memory usage.

        Args:
            batch: the current training batch (after repeat & union with gen_batch_output)
            timing_raw: dict to record probe_gen and probe_reward timings

        Returns:
            probe_scores: Tensor of shape (batch_size, num_truncations + 1) where each
                          value is the fraction of MC samples that got the correct answer.
                          Columns correspond to CoT fractions 0/N, 1/N, ..., N/N (inclusive).
        """
        probe_cfg = self.config.probe
        num_splits = probe_cfg.get('num_splits', 1)  # Default to 1 (no splitting)
        
        batch_size = len(batch)
        
        # If num_splits is 1 or batch_size is too small, run normally
        if num_splits <= 1 or batch_size < num_splits:
            num_splits = 1
        
        tokenizer = self.tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        suffix_str = probe_cfg.suffix

        # Encode boundary tokens (only once)
        suffix_ids = tokenizer.encode(suffix_str, add_special_tokens=False)
        suffix_ids = torch.tensor(suffix_ids, dtype=torch.long)
        think_open_ids = torch.tensor(
            tokenizer.encode("<think>", add_special_tokens=False), dtype=torch.long)
        think_close_ids = torch.tensor(
            tokenizer.encode("</think>", add_special_tokens=False), dtype=torch.long)
        answer_open_ids = torch.tensor(
            tokenizer.encode("<answer>", add_special_tokens=False), dtype=torch.long)

        if num_splits == 1:
            # Original behavior: process entire batch at once
            return self._run_probe_single_batch(
                batch, 0, probe_cfg, tokenizer, pad_token_id, 
                suffix_ids, think_open_ids, think_close_ids, answer_open_ids,
                timing_raw
            )
        
        # Split batch into K sub-batches using chunk method
        assert batch_size % num_splits == 0, f"batch_size ({batch_size}) must be divisible by num_splits ({num_splits})"
        
        num_trunc_points = probe_cfg.num_truncations + 1
        all_probe_scores = []
        
        sub_batch_size = batch_size // num_splits
        print(f'[Probe] Splitting batch of {batch_size} samples into {num_splits} sub-batches of {sub_batch_size} samples each')
        
        # Use chunk method to split the batch
        sub_batches = batch.chunk(chunks=num_splits)
        
        for split_idx in range(num_splits):
            sub_batch = sub_batches[split_idx]
            start_idx = split_idx * sub_batch_size
            
            print(f'[Probe] Processing sub-batch {split_idx + 1}/{num_splits} (samples {start_idx}-{start_idx + sub_batch_size - 1})')
            
            # Process this sub-batch
            sub_probe_scores = self._run_probe_single_batch(
                sub_batch, start_idx, probe_cfg, tokenizer, pad_token_id,
                suffix_ids, think_open_ids, think_close_ids, answer_open_ids,
                timing_raw
            )
            
            all_probe_scores.append(sub_probe_scores)
            
            # Release sub-batch memory
            del sub_batch
            torch.cuda.empty_cache()
            
            print(f'[Probe] Finished split {split_idx + 1}/{num_splits}')
        
        # Concatenate all results
        print(f'[Probe] Concatenating results from all {num_splits} sub-batches...')
        probe_scores = torch.cat(all_probe_scores, dim=0)
        assert probe_scores.shape == (batch_size, num_trunc_points), \
            f"Expected shape ({batch_size}, {num_trunc_points}), got {probe_scores.shape}"
        
        print(f'[Probe] Completed all {num_splits} sub-batches. Final probe_scores shape: {probe_scores.shape}')

        return probe_scores

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                import datetime as _dt
                _now = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'========== Start step {self.global_steps} [{_now}] ==========')
                print(f'Epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # ── PRM Probe (compute only; metrics logged after reward) ──
                    _probe_cfg = getattr(self.config, 'probe', None)
                    if _probe_cfg is not None and _probe_cfg.get('enable', False):
                        probe_scores = self._run_probe(batch, timing_raw)
                        batch.batch['probe_scores'] = probe_scores

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # ── PRM Probe metrics (now that reward is available) ──
                        if 'probe_scores' in batch.batch.keys():
                            probe_scores = batch.batch['probe_scores']
                            num_truncations = self.config.probe.num_truncations
                            num_trunc_points = probe_scores.shape[1]

                            # Per-sample reward: sum token_level_scores along seq dim
                            per_sample_reward = reward_tensor.sum(dim=-1)  # (batch_size,)
                            correct_mask = (per_sample_reward >= 1.0)
                            incorrect_mask = ~correct_mask
                            n_correct = correct_mask.sum().item()
                            n_incorrect = incorrect_mask.sum().item()

                            overconf_weights = torch.linspace(0.5, -0.5, num_trunc_points)

                            def _log_probe_group(prefix, scores):
                                """Log probe metrics for a group of samples."""
                                if scores.shape[0] == 0:
                                    return
                                metrics[f'{prefix}/mean_score'] = scores.mean().item()
                                for k_idx in range(num_trunc_points):
                                    frac = k_idx / num_truncations
                                    metrics[f'{prefix}/score_at_{frac:.2f}'] = scores[:, k_idx].mean().item()
                                oc = (scores * overconf_weights.unsqueeze(0)).sum(dim=1)
                                metrics[f'{prefix}/overconf'] = oc.mean().item()

                            _log_probe_group('probe_all', probe_scores)
                            _log_probe_group('probe_correct', probe_scores[correct_mask])
                            _log_probe_group('probe_incorrect', probe_scores[incorrect_mask])
                            metrics['probe_all/n_correct'] = float(n_correct)
                            metrics['probe_all/n_incorrect'] = float(n_incorrect)

                            # ── Overconfidence penalty on reward ──
                            overconf_coeff = getattr(self.config.probe, 'overconf_coeff', 0.0)
                            if overconf_coeff != 0.0:
                                # Per-sample overconf: weighted sum of probe scores
                                oc_per_sample = (probe_scores * overconf_weights.unsqueeze(0)).sum(dim=1)  # (batch_size,)
                                penalty = overconf_coeff * oc_per_sample  # (batch_size,)

                                # Subtract penalty at the last valid token of each sample
                                # (same position where the original reward is placed)
                                response_length = batch.batch['responses'].shape[-1]
                                attention_mask = batch.batch['attention_mask']
                                response_mask = attention_mask[:, -response_length:]
                                # last valid token index within response for each sample
                                last_valid_idx = response_mask.sum(dim=-1).long() - 1  # (batch_size,)
                                last_valid_idx = last_valid_idx.clamp(min=0)
                                reward_tensor[torch.arange(reward_tensor.shape[0], device=reward_tensor.device), last_valid_idx] -= penalty.to(reward_tensor.device)
                                # Update token_level_scores with the penalised reward
                                batch.batch['token_level_scores'] = reward_tensor

                                metrics['probe_all/overconf_penalty_mean'] = penalty.mean().item()
                                metrics['probe_all/overconf_penalty_abs_mean'] = penalty.abs().mean().item()
                                metrics['probe_all/overconf_coeff'] = overconf_coeff

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # Print metrics to log
                _now_end = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'[Step {self.global_steps}] Metrics [{_now_end}]:')
                for key in sorted(metrics.keys()):
                    print(f'  {key}: {metrics[key]:.6f}' if isinstance(metrics[key], float) else f'  {key}: {metrics[key]}')

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
