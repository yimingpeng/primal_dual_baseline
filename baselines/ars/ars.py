#!/usr/bin/env python3
import inspect
import os
import sys
from collections import deque
from baselines import logger
import time

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), os.pardir)))
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

"""Description:
"""
__author__ = "Yiming Peng"
__copyright__ = "Copyright 2018, baselines"
__credits__ = ["Yiming Peng"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Yiming Peng"
__email__ = "yiming.peng@ecs.vuw.ac.nz"
__status__ = "Prototype"

import os
import sys
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

# Setting the Hyper Parameters
episodes_so_far = 0
timesteps_so_far = 0
iters_so_far = 0
lenbuffer = deque(maxlen = 100)  # rolling buffer for episode lengths
rewbuffer = deque(maxlen = 100)  # rolling buffer for episode rewards


class Hp():
    def __init__(self, main_loop_size, horizon, num_timesteps, step_size, noise):
        self.main_loop_size = main_loop_size
        self.horizon = horizon
        self.max_timesteps = num_timesteps
        self.step_size = step_size
        self.n_directions = 60
        self.b = 20
        assert self.b <= self.n_directions, "b must be <= n_directions"
        self.noise = noise
        self.seed = 1
        ''' chose your favourite '''
        # self.env_name = 'Reacher-v1'
        # self.env_name = 'Pendulum-v0'
        # self.env_name = 'HalfCheetahBulletEnv-v0'
        # self.env_name = 'Hopper-v1'#'HopperBulletEnv-v0'
        # self.env_name = 'Ant-v1'#'AntBulletEnv-v0'#
        # self.env_name = 'InvertedPendulumSwingupBulletEnv-v0'  # 'AntBulletEnv-v0'#
        # self.env_name = 'HalfCheetah-v1'
        # self.env_name = 'Swimmer-v1'
        # self.env_name = 'Humanoid-v1'


# observation filter
class Normalizer():
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


def result_record():
    global lenbuffer, rewbuffer, iters_so_far, timesteps_so_far, \
        episodes_so_far, tstart
    if len(lenbuffer) == 0:
        mean_lenbuffer = 0
    else:
        mean_lenbuffer = np.mean(lenbuffer)
    if len(rewbuffer) == 0:
        # TODO: Add pong game checking
        mean_rewbuffer = 0
    else:
        mean_rewbuffer = np.mean(rewbuffer)
    logger.record_tabular("EpLenMean", mean_lenbuffer)
    logger.record_tabular("EpRewMean", mean_rewbuffer)
    logger.record_tabular("EpisodesSoFar", episodes_so_far)
    logger.record_tabular("TimestepsSoFar", timesteps_so_far)
    logger.record_tabular("TimeElapsed", time.time() - tstart)
    logger.dump_tabular()


# linear policy
class Policy():
    def __init__(self, input_size, output_size, hp):
        self.theta = np.zeros((output_size, input_size))
        self.hp = hp

    def evaluate(self, input):
        return self.theta.dot(input)

    def positive_perturbation(self, input, delta):
        return (self.theta + self.hp.noise * delta).dot(input)

    def negative_perturbation(self, input, delta):
        return (self.theta - self.hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.n_directions)]

    def update(self, rollouts, sigma_r ):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += self.hp.step_size * step / (sigma_r * self.hp.b)


# training loop
def train(env, policy, normalizer, hp):
    global lenbuffer, rewbuffer, iters_so_far, timesteps_so_far, \
        episodes_so_far, tstart
    tstart = time.time()
    rewbuffer.extend(evaluate(env, normalizer, policy))
    # print(rewbuffer)
    result_record()
    record = False
    for episode in range(hp.main_loop_size):
        cur_lrmult = 1.0
        # cur_lrmult = max(1.0 - float(timesteps_so_far) / (0.5 * hp.max_timesteps), 1e-8)
        if timesteps_so_far >= hp.max_timesteps:
            result_record()
            break
        # init deltas and rewards
        deltas = policy.sample_deltas()
        reward_positive = [0] * hp.n_directions
        reward_negative = [0] * hp.n_directions

        record = False

        # positive directions
        for k in range(hp.n_directions):
            state = env.reset()
            done = False
            num_plays = 0.
            while not done and num_plays < hp.horizon:
                normalizer.observe(state)
                state = normalizer.normalize(state)
                action = policy.positive_perturbation(state, deltas[k])
                action = np.clip(action, env.action_space.low, env.action_space.high)
                state, reward, done, _ = env.step(action)
                reward = max(min(reward, 1), -1)
                reward_positive[k] += reward
                num_plays += 1
                timesteps_so_far += 1
                if timesteps_so_far % 10000 == 0 and timesteps_so_far > 0:
                    record = True
            episodes_so_far += 1
            if record:
                # print(total_steps)
                rewbuffer.extend(evaluate(env, normalizer, policy))
                # print(rewbuffer)
                # print("Averge Rewards:", np.mean(rewbuffer))
                result_record()
                record = False
        # negative directions
        for k in range(hp.n_directions):
            state = env.reset()
            done = False
            num_plays = 0.
            while not done and num_plays < hp.horizon:
                normalizer.observe(state)
                state = normalizer.normalize(state)
                action = policy.negative_perturbation(state, deltas[k])
                state, reward, done, _ = env.step(action)
                reward = max(min(reward, 1), -1)
                reward_negative[k] += reward
                num_plays += 1
                timesteps_so_far += 1
                if timesteps_so_far % 10000 == 0 and timesteps_so_far > 0:
                    record = True
            episodes_so_far += 1
            if record:
                # print(total_steps)
                # print(rewbuffer)
                rewbuffer.extend(evaluate(env, normalizer, policy))
                # print("Averge Rewards:", np.mean(rewbuffer))
                result_record()
                record = False
        all_rewards = np.array(reward_negative + reward_positive)
        sigma_r = all_rewards.std()

        # sort rollouts wrt max(r_pos, r_neg) and take (hp.b) best
        scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(reward_positive, reward_negative))}
        order = sorted(scores.keys(), key = lambda x: scores[x])[-hp.b:]
        rollouts = [(reward_positive[k], reward_negative[k], deltas[k]) for k in order[::-1]]

        hp.step_size = hp.step_size * cur_lrmult
        # update policy:
        policy.update(rollouts, sigma_r)

        # evaluate

        # finish, print:
        # print('episode',episode,'reward_evaluation',reward_evaluation)


def evaluate(env, normalizer, policy):
    state = env.reset()
    num_plays = 0
    reward_evaluation = 0
    ep_num = 0
    rewards = []
    while True:
        if ep_num >= 5:
            break
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state)
        state, reward, done, _ = env.step(action)
        reward_evaluation += reward
        num_plays += 1
        if done:
            ep_num += 1
            rewards.append(reward_evaluation)
            reward_evaluation = 0
            num_plays = 0
            state = env.reset()
    return rewards


if __name__ == '__main__':
    hp = Hp()
    np.random.seed(hp.seed)
    env = gym.make(hp.env_name)
    # env = wrappers.Monitor(env, monitor_dir, force=True)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    policy = Policy(num_inputs, num_outputs, hp)
    normalizer = Normalizer(num_inputs)
    train(env, policy, normalizer, hp)
