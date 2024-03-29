#!/usr/bin/env python

# ------------- import everything you need -------------
import gym
import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import cv2
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# ------------------------------------------------------

# ------------ global variables -----------------------
# if gpu is available
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
Tensor = FloatTensor

# namedtuple for storing experiences
Transition = namedtuple(
    'Transition', ('observations', 'actions', 'rewards', 'dones',
                   'next_observations')
)

EPS = 3e-3
# -------------------------------------------------------------


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


# ---------------- experience replay buffer -------------------
class Buffer(object):
    def __init__(self, max_length=int(1e6)):
        self.max_length = max_length
        self.data = []
        self.index = 0

    # add a sample to the buffer
    def add_sample(self, *args):
        if len(self.data) < self.max_length:
            self.data.append(None)
        self.data[self.index] = Transition(*args)
        self.index = (self.index + 1) % self.max_length

    # random sampling
    def sample(self, batch_size):
        return random.sample(self.data, batch_size)

    # manually clear data (required in cases when exp. replay is not used)
    def clear(self):
        self.data = []
        self.index = 0

    # used again in cases when exp. replay is not used
    def get_all_data(self):
        return self.data

    # length of data buffer
    def __len__(self):
        return len(self.data)


def save_checkpoint(state, filename):
    torch.save(state, filename)


class DQNmodel(object):
    def __init__(self, env, gamma):
        self.env = env
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]

        self.main = DDQN(self.s_dim, self.a_dim)
        self.target = DDQN(self.s_dim, self.a_dim)
        if use_cuda:
            self.main.cuda()
            self.target.cuda()

        self.memory = Buffer(max_length=25*int(1e4))
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main.parameters(), lr=2.5*1e-4)
        self.tau = 0.001
        hard_update(self.target, self.main)

    def train(self, batch_size=None):
        if batch_size is None:
            transition = self.memory.get_all_data()
        else:
            transition = self.memory.sample(batch_size)

        batch = Transition(*zip(*transition))
        batch_obs = torch.cat(batch.observations, dim=0)
        batch_next_obs = torch.cat(batch.next_observations, dim=0)
        batch_a = batch.actions
        batch_r = Variable(Tensor(batch.rewards)).type(FloatTensor)
        batch_done = Variable(Tensor(batch.dones)).type(FloatTensor)

        inp = Variable(batch_next_obs, volatile=True).type(FloatTensor)
        Q1, _ = torch.max(self.target.forward(inp), dim=1)
        y = batch_r + self.gamma * (1-batch_done) * Q1

        ss = Variable(batch_obs).type(FloatTensor)
        Q_pred = self.main.forward(ss)[range(batch_size), list(batch_a)]

        loss = self.criterion(Q_pred, y)
        #print (loss.data[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.target, self.main, self.tau)

    def test(self, nepisodes):
        total_reward = 0
        eps = .05
        episode_max_length = self.env._max_episode.length
        for _ in range(nepisodes):
            episode_reward = 0
            s = self.env.reset()
            for _ in range(episode_max_length):
                action = self.policy(s, eps=eps)
                s1, r, done = self.env.step(action)
                episode_reward += r
                s = s1[:]
                if done:
                    break
            total_reward += episode_reward
        return total_reward

    def add_to_memory(self, obs, action, reward, done, next_obs):
        s = torch.unsqueeze(torch.from_numpy(np.array(obs)), 0)
        s1 = torch.unsqueeze(torch.from_numpy(np.array(next_obs)), 0)
        self.memory.add_sample(s, action, reward, done*1.0, s1)

    # TODO: supports only one input vector for now (not sure though)
    def policy(self, x, eps=.05):
        if np.random.rand(1) < eps:
            return self.env.action_space.sample()
        else:
            obs = torch.unsqueeze(torch.from_numpy(np.array(x)), 0)
            inp = Variable(obs, volatile=True).type(FloatTensor)
            q_values = self.main.forward(inp).data
            _, a = torch.max(q_values, 1)
            return a[0]

    def save(self, episode_num, filename):
        save_checkpoint({
            'episode_number': episode_num,
            'state_dict': self.main.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename=filename)


class DQN(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(DQN, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = 3136
        self.h2_dim = 512

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16,
                               kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1)
        self.l1 = nn.Linear(self.h1_dim, self.h2_dim)
        self.l2 = nn.Linear(self.h2_dim, self.a_dim)

        nn.init.xavier_normal(self.conv1.weight.data)
        nn.init.xavier_normal(self.conv2.weight.data)
        nn.init.xavier_normal(self.conv3.weight.data)
        nn.init.xavier_normal(self.l1.weight.data)
        nn.init.uniform(self.l2.weight.data, a=-EPS, b=EPS)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.l1(y.view(-1, self.h1_dim)))
        y = self.l2(y)
        return y


# ----------------------------------------------------------------------

# ------------------ Dueling DQN --------------------------------------

# stands for Dueling DQN (and not Double DQN)
class DDQN(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(DDQN, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = 3136
        self.h2_dim = 512

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16,
                               kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1)
        self.l1 = nn.Linear(self.h1_dim, self.h2_dim)
        self.advantage = nn.Linear(self.h2_dim, self.a_dim)
        self.value = nn.Linear(self.h2_dim, 1)

        nn.init.xavier_normal(self.conv1.weight.data)
        nn.init.xavier_normal(self.conv2.weight.data)
        nn.init.xavier_normal(self.conv3.weight.data)
        nn.init.xavier_normal(self.l1.weight.data)
        nn.init.uniform(self.advantage.weight.data, a=-EPS, b=EPS)
        nn.init.uniform(self.value.weight.data, a=-EPS, b=EPS)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.l1(y.view(-1, self.h1_dim)))
        adv = self.advantage(y)
        value = self.value(y)
        return adv + value


def preprocess_image(s):
    gray = np.dot(s[..., :3], [.299, .587, .114])
    img = cv2.resize(gray, (84, 84))
    img = img/255.0
    return img


if __name__ == '__main__':
    # gamma depends on the environment, so this setup
    env_names = ['MountainCar-v0', 'CartPole-v0']
    gammas = [1, .99]
    # 0 - mountaincar || 1 - cartpole
    env_id = 0
    gamma = gammas[env_id]
    env_name = env_names[env_id]
    #env = gym.make(env_name)


    env = gym.make('SpaceInvaders-v0')
    # mdl = QN_linear(s_dim, a_dim, gamma)
    mdl = DQNmodel(env, gamma=.99)
    # mdl = DDQN(s_dim, a_dim, gamma)


    nepisodes = 10000
    episode_max_length = env._max_episode_steps + 1
    batch_size = 64
    stack_size = 4
    update_frequency = 4*stack_size
    e = 1.0
    e_min = .05
    # set it to 2*10^5 for mountain car and 10^4 for cartpole
    # duelling DQN rocks even for 10^4 in mountain car
    annealing_steps = int(1e6)      # one million frames
    e_step = (e - e_min) / annealing_steps

    i = 0
    steps = 0
    render = False
    rall = []
    action_freq = 3
    image_stack = []
    pre_train_steps = batch_size + stack_size
    #pre_train_steps = 5*int(1e4)
    save_freq = 100

    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)

    while i < nepisodes:
        i += 1
        s = env.reset()
        s = preprocess_image(s)

        '''
        for _ in range(1000):
            a = env.action_space.sample()
            s1, r, done, _ = env.step(a)
            s1 = preprocess_image(s1)
            s = s1[:]
            env.render()
        '''

        j = 0
        episode_reward = 0
        # todo: initial frame should be the first frame or empty frame ?

        image_stack = []
        for _ in range(stack_size):
            image_stack.append(s)

        while j < episode_max_length:
            j += 1
            steps += 1
            if j % action_freq == 1:
                action = mdl.policy(image_stack, eps=e)

            e = max(e_min, e - e_step)
            s1, r, done, _ = env.step(action)
            #print (i, j, r, done)
            s1 = preprocess_image(s1)

            next_image_stack = image_stack[:]
            next_image_stack.append(s1)
            next_image_stack.pop(0)

            rwd = r/abs(r) if r != 0 else 0
            mdl.add_to_memory(image_stack, action, rwd, done, next_image_stack)
            image_stack = next_image_stack[:]

            if render:
                env.render()

            s = s1[:]
            episode_reward += r

            if steps > pre_train_steps and steps % update_frequency == 0:
                mdl.train(batch_size)

            if done:
                # mdl.train()
                # mdl.clear_memory()
                break

        rall.append(episode_reward)
        print ('Episode: %3d, e: %.4f, Reward: %3d, Avg Reward: %.2f'
               % (i, e, episode_reward, np.mean(rall[-100:])))

        if i % save_freq == 0:
            filename = directory + '/episode_' + str(i) + '.pt'
            mdl.save(i, filename)

