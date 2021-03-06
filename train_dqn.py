import os, random, time, argparse, gym, sys
import json
import os.path as osp, shutil, time, atexit, os, subprocess
import logz
import numpy as np
import torch
import torchvision
import torch.nn as nn
import dqn
from dqn_utils import *
from gym import wrappers
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
    # limit = max(int(args.num_steps/2), 2e6)
    # exploration_schedule = PiecewiseSchedule([
    #         (0,     1.00),
    #         (1e6,   0.10),
    #         (limit, 0.01),
    #     ], outside_value=0.01
    # )
    three_fourths = 3*args.num_steps/4
    seven_eigths = 7*args.num_steps/8
    exploration_schedule = PiecewiseSchedule([
            (0,     1.00),
            (three_fourths,   0.10),
            (seven_eigths, 0.01),
        ], outside_value=0.01
    )
    policy = dqn.learn(
        env=env,
        q_func_model=base_atari_model,
        exploration=exploration_schedule,
        replay_buffer_size=args.replay_buffer_size,
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
        pr_beta=args.pr_beta,
        pr_alpha=args.pr_alpha,
        n=args.n,
        h=args.h,
        icm=args.icm,
        icm_gamma=args.icm_gamma,
        icm_beta=args.icm_beta,
        icm_eta=args.icm_eta
    )
    env.close()
    return policy


def set_global_seeds(i):
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_env(args):

    print("the arguments of gym.make")
    import inspect
    print(inspect.getargspec(gym.make))

    env = gym.make("procgen:procgen-" + args.env + "-v0", num_levels=args.num_levels, start_level=args.start_seed, distribution_mode='easy')
    # env = gym.make("procgen:procgen-" + args.env + "-v0", args)

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
    parser.add_argument('--n', type = int, default = 1)
    parser.add_argument('--h', action='store_true', default=False)
    parser.add_argument('--replay_buffer_size', default=1000000)
    # parser.add_argument('--num_envs', type=int, default=1) # can be used for parallel agents?
    parser.add_argument('--root_logdir', default='./data_dqn')

    # for Prioritized Replay
    parser.add_argument('--pr', action='store_true', default=False)
    parser.add_argument('--pr_beta', type=float, default=0.4)
    parser.add_argument('--pr_alpha', type=float, default=0.6)

    # for ICM
    parser.add_argument('--icm', action='store_true', default=False)
    parser.add_argument('--icm_gamma', type=float, default=0.1)
    parser.add_argument('--icm_beta', type=float, default=0.2)
    parser.add_argument('--icm_eta', type=float, default=10)

    parser.add_argument('--run_test_num', type=int, default=0)
    args = parser.parse_args()

    assert args.n >= 1, "n-step must be at least 1."
    assert args.env in ['coinrun', 'caveflyer', 'jumper', 'fruitbot']
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    print('random seed = {}'.format(args.seed))
    exp_name = 'dqn'
    if args.double_q:
        exp_name = 'double-dqn'
    if args.pr:
        exp_name += '_pr'
    if args.icm:
        exp_name += '_icm'

    if args.n:
        print("warning: do not use --n.")

    if not(os.path.exists(args.root_logdir)):
        os.makedirs(args.root_logdir)

    logdir = exp_name+ '_' +args.env+ '_' +time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(args.root_logdir, logdir)
    logz.configure_output_dir(logdir)
    args.logdir = logdir

    logz.save_params(vars(args), args.logdir)

    env = get_env(args)
    policy = learn(env, args)
    
    result = {}
    for i in range(args.run_test_num):
        seed = random.randint(0, 999999)
        env = gym.make("procgen:procgen-" + args.env + "-v0", num_levels=1, start_level=seed, distribution_mode='easy')
        env.seed(seed)
        #print("Run with seed " + str(seed) + ": " + str(dqn.step_best(env, policy)))
        result[seed] = dqn.step_best(env, policy)
        env.close()
    
    with open(osp.join(logdir, "testing_results.json"), 'w') as out:
        out.write(json.dumps(result, separators=(',\n','\t:\t'), sort_keys=True))
