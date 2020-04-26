import os, random, time, argparse, gym, sys
import logz
from gym import wrappers
import numpy as np
import torch
import torchvision
import torch.nn as nn
import dqn
from dqn_utils import *
# from procgen import ProcgenEnv

class base_atari_model(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(base_atari_model, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.action_value = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_dim)
        )
    def forward(self, x):
        x = self.convnet(x)
        x = x.reshape(x.shape[0], -1)
        x = self.action_value(x)
        return x

def learn(env, args):
    # lr_schedule = ConstantSchedule(1e-4)
    limit = max(int(args.num_steps/2), 2e6)
    exploration_schedule = PiecewiseSchedule([
            (0,     1.00),
            (1e6,   0.10),
            (limit, 0.01),
        ], outside_value=0.01
    )
    dqn.learn(
        env=env,
        q_func_model=base_atari_model,
        exploration=exploration_schedule,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        logdir=args.logdir,
        max_steps=args.num_steps,
        double_q=args.double_q,
        pr=args.pr,
        beta=args.beta,
        alpha=args.alpha
    )
    env.close()


def set_global_seeds(i):
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_env(args):
    env = gym.make("procgen:procgen-" + args.env + "-v0", num_levels=args.num_levels, start_level=args.start_seed, distribution_mode='easy')
    set_global_seeds(args.seed)
    env.seed(args.seed)
    expt_dir = os.path.join(args.logdir, "procgen")
    return wrappers.Monitor(env, expt_dir, force=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--num_steps', type=int, default=1e5) #100K, 250K, or 1M
    parser.add_argument('--double_q', action='store_true', default=False)
    parser.add_argument('--num_levels', type=int, default=50) #50, 100, 250, 500
    parser.add_argument('--pr', action='store_true', default=False)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=0.6)
    # parser.add_argument('--num_envs', type=int, default=1) # can be used for parallel agents?
    args = parser.parse_args()

    assert args.env in ['coinrun', 'caveflyer', 'jumper']
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    print('random seed = {}'.format(args.seed))
    exp_name = 'dqn'
    if args.double_q:
        exp_name = 'double-dqn'
    if args.pr:
        exp_name += '_pr'

    if not(os.path.exists('data_dqn')):
        os.makedirs('data_dqn')
    logdir = exp_name+ '_' +args.env+ '_' +time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data_dqn', logdir)
    logz.configure_output_dir(logdir)
    args.logdir = logdir
    env = get_env(args)
    learn(env, args)
