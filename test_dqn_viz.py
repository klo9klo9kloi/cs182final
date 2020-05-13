import json
import torch
import torchvision
import os, random, time, argparse, gym, sys
import os.path as osp
import dqn
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import skimage.io as skio
import numpy as np
import glob
from tqdm import tqdm
from train_dqn import base_atari_model, set_global_seeds
from matplotlib import animation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--logdir', default='./data_dqn')
    parser.add_argument('--frame_history_len', type=int, default=4)
    parser.add_argument('--start_seed', type=int, default=500)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--animate', action='store_true', default=False)
    args = parser.parse_args()

    assert args.env in ['coinrun', 'caveflyer', 'jumper', 'fruitbot']

    env = gym.make("procgen:procgen-" + args.env + "-v0", num_levels=1, distribution_mode='easy')
    num_actions = env.action_space.n
    env.close()

    policy = base_atari_model(args.frame_history_len * 3, num_actions)
    load_path = osp.join(args.logdir, 'policy.pth')

    assert osp.exists(load_path), "no trained policy detected, please verify your logdir path"

    policy.load_state_dict(torch.load(load_path))
    policy.to(device)

    result = {}
    if args.seed < 0:
        seed = random.randint(0, 999999)
        set_global_seeds(seed)
    else:
        seed = args.seed

    env = gym.make("procgen:procgen-" + args.env + "-v0", num_levels=1, start_level=args.start_seed, distribution_mode='easy')
    env.seed(seed) 
  
    # Uses the given policy to step the env until done. Used to evaluate the trained policy.
    obs = env.reset()
    frames = [np.zeros_like(obs) for _ in range(args.frame_history_len - 1)]
    frames.append(obs)
    encoded_obs = np.concatenate(frames, 2)

    done = False

    frames_path = osp.join(args.logdir, 'viz_frames')
    if not osp.exists(frames_path):
        os.makedirs(frames_path)

    policy.eval()
    i = 0
    y_max = 5
    while (not done):
        policy.zero_grad()
        encoded_obs1 = torchvision.transforms.functional.to_tensor(encoded_obs)
        encoded_obs1 = encoded_obs1.reshape([1] + list(encoded_obs1.shape)).to(device)

        encoded_obs1 = torch.autograd.Variable(encoded_obs1, requires_grad=True)
        q_vals = policy(encoded_obs1)
        q_vals_numpy = q_vals.cpu().detach().numpy()
        action = np.argmax(q_vals_numpy)

        action_value = q_vals.squeeze()[action]
        action_value.backward()

        saliency = torch.max(torch.abs(encoded_obs1.grad), dim=1).values

        gs = gridspec.GridSpec(4, 4)
        plt.figure()
        ax = plt.subplot(gs[0:2, 0:2])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(obs)
        ax = plt.subplot(gs[0:2, 2:])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(saliency.cpu().detach().numpy().reshape(64, 64), cmap=plt.cm.hot)
        ax = plt.subplot(gs[2:, :])
        plt.bar(np.arange(num_actions), q_vals_numpy.squeeze())
        plt.xlabel("Action")
        plt.ylabel("Q-val")
        plt.ylim(top=y_max)
        # plt.show()
        plt.savefig(osp.join(frames_path, str(i) + ".png"))
        plt.close()

        obs, rew, done, info = env.step(action)
        encoded_obs = np.append(encoded_obs[:, :, 3:], obs, 2)
        i+=1

    env.close()
    
    if args.animate:
        filenames = sorted(glob.glob(osp.join(frames_path, "*.png")), key=lambda x: int(x.split('/')[-1].split('.')[0]))

        imgs = [plt.imread(f) for f in filenames]
        fig = plt.figure()
        im = plt.imshow(np.zeros((640, 640, 3))) #parametrize img shape

        def init():
            im.set_data(imgs[0])
            return [im]
        def animate(i):
            im.set_data(imgs[i])
            return [im]

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(imgs), blit=True) #parametrize fps
        anim.save(os.path.join(frames_path, 'anim.mp4'), fps=27, extra_args=['-vcodec', 'libx264'])
