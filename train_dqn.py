import os, random, time, argparse, gym, sys
import logz
from gym import wrappers
import numpy as np
import torch
import dqn
from dqn_utils import *
from models.models import *
# from procgen import ProcgenEnv

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
    dqn.learn(
        env=env,
        q_func_model=model_with_glimpse if args.attention else base_atari_model,
        exploration=exploration_schedule,
        replay_buffer_size=500000,
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
        alpha=args.alpha,
        n=args.n,
        h=args.h
    )
    env.close()


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
    # import pdb; pdb.set_trace()
    
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
    parser.add_argument('--n', type = int, default = 1)
    parser.add_argument('--h', action='store_true', default=False)
    # parser.add_argument('--num_envs', type=int, default=1) # can be used for parallel agents?
    parser.add_argument('--root_logdir', default='./data_dqn')
    parser.add_argument('--attention', action='store_true', default=False)
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
    if args.attention:
        exp_name += '_attention'

    if not(os.path.exists(args.root_logdir)):
        os.makedirs(args.root_logdir)
    logdir = exp_name+ '_' +args.env+ '_' +time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(args.root_logdir, logdir)
    logz.configure_output_dir(logdir)
    args.logdir = logdir
    
    logz.save_params(vars(args), args.logdir)

    env = get_env(args)
    learn(env, args)
