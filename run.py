import numpy as np
import matplotlib.pyplot as plt
from procgen import ProcgenEnv
from matplotlib import animation


# from agents import bestagent420
# env = gym.make('procgen:procgen-coinrun-v0', distribution_mode="easy", num_levels=50)
env = ProcgenEnv(num_envs=1, env_name='coinrun', num_levels=50, start_level=0, distribution_mode='easy')
obs = env.reset()
step = 0
# env.action_space = Discrete(n)
# possible actions = env.action_space.

observations = []
observations.append(obs['rgb'][0])
fig = plt.figure()
im = plt.imshow(np.zeros(observations[0].shape))

def init():
    im.set_data(np.zeros(observations[0].shape))
    return [im]

def animate(i):
    im.set_data(observations[i])
    return [im]

levels_completed = 0
while True:
    obs, rew, done, info = env.step(np.array(env.action_space.sample())) # our task: create an agent that will generate an action at each step given the obs and reward
    print(f"step {step} reward {rew} done {done}")
    observations.append(obs['rgb'][0])
    step += 1
    if done:
        env.reset()
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(observations), interval=25, blit=True)
        plt.show()

        fig=plt.figure()
        observations = []
        observations.append(np.zeros((64,64,3)))
        im = plt.imshow(np.zeros((64,64,3)))


