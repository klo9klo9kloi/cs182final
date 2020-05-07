import json
import torch
import os
import os.path as osp, shutil, time, atexit, os, subprocess
import os, random, time, argparse, gym, sys
import dqn
from train_dqn import base_atari_model, set_global_seeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--logdir', default='./data_dqn')
    parser.add_argument('--run_test_num', type=int, default=0)
    parser.add_argument('--frame_history_len', type=int, default=4)
    args = parser.parse_args()

    assert args.env in ['coinrun', 'caveflyer', 'jumper', 'fruitbot']
    num_actions = env.action_space.n
    policy = base_atari_model(args.frame_history_len * 3, num_actions)
    load_path = osp.join(args.logdir, 'policy.pth')

    assert osp.exists(load_path), "no trained policy detected, please verify your logdir path"

    policy.load_state_dict(torch.load(load_path))

    result = {}
    for i in range(args.run_test_num):
        seed = random.randint(0, 999999)
        set_global_seeds(seed)
        env = gym.make("procgen:procgen-" + args.env + "-v0", num_levels=1, start_level=seed, distribution_mode='easy')
        env.seed(seed)
        #print("Run with seed " + str(seed) + ": " + str(dqn.step_best(env, policy)))
        result[seed] = dqn.step_best(env, policy)
        env.close()
    
    with open(osp.join(args.logdir, "testing_results.json"), 'w') as out:
        out.write(json.dumps(result, separators=(',\n','\t:\t'), sort_keys=True))
