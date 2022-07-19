import gym
import torch
import numpy as np
from copy import deepcopy
from torch.multiprocessing import Process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: Implement this
def cat_values(obs):
    """
    Concatenate the values of the NamedTuple/Observation/Dictionary state
    """
    raise NotImplementedError()
    return np.zeros([2, 3])


class EnvWorker(Process):
    def __init__(self, env_name, render, child_conn):
        super(EnvWorker, self).__init__()
        self.env = gym.make(env_name)
        self.render = render
        self.child_conn = child_conn
        self.init_state()

    def init_state(self):
        obs = self.env.reset()
        
        obs, _, _, _ = self.env.step(1)
        obs = cat_values(obs)
        self.history = np.moveaxis(obs, -1, 0)

    def run(self):
        super(EnvWorker, self).run()

        episode = 0
        steps = 0
        total_reward = 0
        crashed = False

        while True:
            if self.render:
                self.env.render()

            action = self.child_conn.recv()
            next_obs, reward, done, info = self.env.step(action + 1)

            # TODO: pre-process next_obs to see if vehicle collided
            if False:  # TODO: replace here
                crashed = True

            next_obs = cat_values(next_obs)
            self.history = np.moveaxis(next_obs, -1, 0)

            steps += 1
            total_reward += reward

            self.child_conn.send([deepcopy(self.history), reward, crashed, done])

            if done or crashed:
                # print('{} episode | total_reward: {:2f} | steps: {}'.format(
                #     episode, total_reward, steps
                # ))
                episode += 1
                steps = 0
                total_reward = 0
                crashed = False
                self.init_state()