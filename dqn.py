import uuid
import time
import pickle
import os
import sys
import gym.spaces
import itertools
import numpy as np
import random
import logz
import torch
import torch.nn as nn
import torchvision
from collections import namedtuple
from dqn_utils import *
from icm.curiosity import CuriosityModule

# DQNSpec = namedtuple("DQNSpec", ["constructor", "kwargs", "lr_schedule"])
# PRSpec = namedtuple("PRSpec", ["alpha", "beta"])
# ICMSpec = namedtuple("ICMSpec", ["module", "optimizer", "history"])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def huber_loss(x, delta=1.0):
    """https://en.wikipedia.org/wiki/Huber_loss
    """
    return torch.where(
        torch.abs(x) < delta,
        torch.pow(x, 2) * 0.5,
        delta * (torch.abs(x) - 0.5 * delta)
    )


def step_env(env, replay_buffer, num_actions, exploration_schedule, t, last_obs, model_initialized, policy, n, gamma, icm):
    frame_idx = replay_buffer.store_frame(last_obs)
    encoded_obs = replay_buffer.encode_recent_observation()
    explor_prob = exploration_schedule.value(t)

    first_action = -1
    first_done = False
    first_obs = None
    reward = 0

    for i in range(n):
        if not model_initialized or (not icm and np.random.random() < explor_prob):
            action = np.random.randint(num_actions)
        else:
            with torch.no_grad():
                policy.eval()
                encoded_obs = torchvision.transforms.functional.to_tensor(encoded_obs)
                encoded_obs = encoded_obs.reshape([1] + list(encoded_obs.shape)).to(device)
                q_vals = policy(encoded_obs)
            action = np.argmax(q_vals.cpu().numpy())

        if i == 0:
            first_action = action

        encoded_obs, rew, done, info = env.step(action)
        if i == 0:
            first_obs = encoded_obs
            first_done = done

        reward += rew * pow(gamma, i)

        if done:
            replay_buffer.store_effect(frame_idx, first_action, reward, first_done)
            if i == 0:
                first_obs = env.reset()
            else:
                env.reset()
            return first_obs

    replay_buffer.store_effect(frame_idx, first_action, reward, first_done)
    return first_obs

def hf(x, epsilon = 0.0001):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x

def h_inv(x, epsilon = 0.0001):
    return torch.sign(x) * (1 + torch.abs(x - torch.sign(x) * x * epsilon)) ** 2 - 1

def update_model(optimizer, t, replay_buffer, policy, target, gamma, clip, batch_size, num_actions,
                    learning_starts, learning_freq, num_param_updates, target_update_freq, double_q, pr, beta, h, icm_module, icm_optimizer, icm_r_history):
    if (t > learning_starts and \
        t % learning_freq == 0 and \
        replay_buffer.can_sample(batch_size)):

        sample = replay_buffer.sample(batch_size)
        if pr:
            transition_infos, priorities, indices = sample
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = transition_infos
        else:
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = sample

        obs_batch = torch.stack([torchvision.transforms.functional.to_tensor(ob) for ob in obs_batch]).to(device)
        next_obs_batch = torch.stack([torchvision.transforms.functional.to_tensor(ob) for ob in next_obs_batch]).to(device)
        act_batch = torch.from_numpy(act_batch).type(torch.LongTensor).to(device)
        rew_batch = torch.from_numpy(rew_batch).to(device)
        done_mask = torch.from_numpy(done_mask).to(device)

        with torch.no_grad():
            q_target_tp1 = target(next_obs_batch)

        policy.train()
        optimizer.zero_grad()
        policy.zero_grad()

        q_t = policy(obs_batch).gather(1, act_batch.reshape(-1, 1)).flatten()

        if double_q:
            with torch.no_grad():
                policy.eval()
                q_tp1 = torch.argmax(policy(next_obs_batch), axis=1)
                if h:
                    y_t = hf(rew_batch + (1-done_mask) * gamma * h_inv(q_target_tp1.gather(1, q_tp1.reshape(-1, 1)).flatten()))
                else:
                    y_t = rew_batch + (1-done_mask)*gamma*q_target_tp1.gather(1, q_tp1.reshape(-1, 1)).flatten()
        else:
            with torch.no_grad():
                if h:
                    y_t = hf(rew_batch + (1-done_mask) * gamma * h_inv(torch.max(q_target_tp1, axis=1)[0].reshape(-1, 1).flatten()))
                else:
                    y_t = rew_batch + (1-done_mask)*gamma*torch.max(q_target_tp1, axis=1)[0].reshape(-1, 1).flatten()

        if icm_module is not None:
            icm_optimizer.zero_grad()
            # add intrinsic reward to target
            icm_loss, y_t_intrinsic = icm_module(obs_batch, next_obs_batch, act_batch)
            icm_r_history.append(np.mean(y_t_intrinsic.data.cpu().numpy())) # for logging
            y_t += y_t_intrinsic

        if pr:
            is_weights = (priorities*replay_buffer.size + 1e-10)**-beta
            is_weights /= np.max(is_weights)
            is_weights = torch.from_numpy(is_weights).to(device)
            total_error = torch.mean(is_weights * huber_loss(y_t - q_t))
        else:
            total_error = torch.mean(huber_loss(y_t - q_t))

        if icm_module is not None:
            # add icm loss and scale Q-function estimation loss
            total_error = icm_module.gamma * total_error + icm_loss

        total_error.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), clip)
        optimizer.step()

        if icm_module is not None:
            icm_optimizer.step()

        if pr:
            replay_buffer.update_priorities(np.abs(y_t.data.cpu().numpy()), indices)

        if num_param_updates % target_update_freq == 0:
            target.load_state_dict(policy.state_dict())
        num_param_updates += 1
    return num_param_updates


def log_progress(env, t, log_every_n_steps, lr, start_time, exploration, best_mean_episode_reward, icm_r_history):
    episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()

    if len(episode_rewards) > 0:
        mean_episode_reward = np.mean(episode_rewards[-100:])
        std_episode_reward  = np.std(episode_rewards[-100:])
    if len(icm_r_history) > 100:
        mean_icm_r = np.mean(icm_r_history[-100:])
    else:
        mean_icm_r = np.nan
    if len(episode_rewards) > 100:
        best_mean_episode_reward = \
            max(best_mean_episode_reward, mean_episode_reward)

    # See the `log.txt` file for where these statistics are stored.
    if t % log_every_n_steps == 0:
        hours = (time.time() - start_time) / (60.*60.)
        logz.log_tabular("Steps",                 t)
        logz.log_tabular("Avg_Last_100_Episodes", mean_episode_reward)
        logz.log_tabular("Std_Last_100_Episodes", std_episode_reward)
        logz.log_tabular("Best_Avg_100_Episodes", best_mean_episode_reward)
        logz.log_tabular("Num_Episodes",          len(episode_rewards))
        logz.log_tabular("Exploration_Epsilon",   exploration.value(t))
        logz.log_tabular("Adam_Learning_Rate",    lr)
        logz.log_tabular("Avg_IntrReward_Last_100_Episodes", mean_icm_r)
        logz.log_tabular("Elapsed_Time_Hours",    hours)
        logz.dump_tabular()
    return best_mean_episode_reward


def learn(env,
         q_func_model,
         exploration,
         replay_buffer_size,
         batch_size,
         gamma,
         learning_starts,
         learning_freq,
         frame_history_len,
         target_update_freq,
         grad_norm_clipping,
         logdir=None,
         max_steps=2e8,
         double_q=True,
         pr=False,
         pr_beta=0.4,
         pr_alpha=0.6,
         n=1,
         h=False,
         icm=False,
         icm_gamma=0.1,
         icm_beta=0.2,
         icm_eta=10):

    num_actions = env.action_space.n
    policy = q_func_model(3 * frame_history_len, num_actions).to(device)
    target = q_func_model(3 * frame_history_len, num_actions).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    if pr:
        replay_buffer = PrioritizedReplayBuffer(replay_buffer_size, frame_history_len, pr_alpha)
    else:
        replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
    replay_buffer_idx = None

    icm_module = None
    icm_optimizer = None
    if icm:
        # we know ProcGen frames are all 64 x 64
        icm_module = CuriosityModule((3 * frame_history_len, 64, 64), 32, 4, 256, 256, num_actions, device, gamma=icm_gamma, beta=icm_beta, reward_multiplier=icm_eta).to(device)
        icm_optimizer = torch.optim.Adam(icm_module.parameters(), lr=1e-3)

    start_time = time.time()
    log_every_n_steps = 10000

    t = 0
    num_param_updates = 0
    best_mean_episode_reward = float('-inf')
    icm_r_history = []

    last_obs = env.reset()

    while True:
        last_obs = step_env(env, replay_buffer, num_actions, exploration, t, last_obs, t > learning_starts, policy, n, gamma, icm)

        num_param_updates = update_model(optimizer, t, replay_buffer, policy, target, gamma, grad_norm_clipping, batch_size, num_actions,
                    learning_starts, learning_freq, num_param_updates, target_update_freq, double_q, pr, pr_beta, h, icm_module, icm_optimizer, icm_r_history)
        t += 1
        log_progress(env, t, log_every_n_steps, optimizer.param_groups[0]['lr'], start_time, exploration, best_mean_episode_reward, icm_r_history)
        if t > max_steps:
            print("\nt = {} exceeds max_steps = {}".format(t, max_steps))
            sys.exit()
