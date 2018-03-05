#!/usr/bin/env python

# ------------- import everything you need -------------
import gym
import numpy as np
import random
from collections import namedtuple

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
# -------------------------------------------------------------

# ---------------- experience replay buffer -------------------


class Buffer(object):
    def __init__(self, max_length = int(1e6)):
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
# ------------------------------------------------------------------


class QN_linear(nn.Module):
    def __init__(self, s_dim, a_dim, gamma):
        super(QN_linear, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = 128

        self.l1 = nn.Linear(self.s_dim, self.h1_dim)
        self.l2 = nn.Linear(self.h1_dim, self.a_dim)

        self.memory = Buffer()
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return y

    # TODO: supports only one input vector for now
    def policy(self, x, eps=.05):
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.a_dim)
        else:
            obs = torch.from_numpy(x).view(-1, self.s_dim)
            inp = Variable(obs, volatile = True).type(FloatTensor)
            Qvals = self.forward(inp).data
            _, a = torch.max(Qvals, 1)
            return a[0]

    def train(self, batch_size=None):
        if batch_size is None:
            transition = self.memory.get_all_data()
        else:
            transition = self.memory.sample(batch_size)

        batch = Transition(*zip(*transition))
        batch_obs = torch.cat(batch.observations, dim=0)
        batch_next_obs = torch.cat(batch.next_observations, dim=0)
        batch_a = batch.actions
        # todo: not sure if volatile should be set to true
        batch_r = Variable(Tensor(batch.rewards)).type(FloatTensor)
        batch_done = Variable(Tensor(batch.dones)).type(FloatTensor)

        inp = Variable(batch_next_obs, volatile=True).type(FloatTensor)
        q1, _ = torch.max(self.forward(inp), dim=1)
        y = batch_r + self.gamma * (1-batch_done) * q1

        ss = Variable(batch_obs).type(FloatTensor)
        q_pred = self.forward(ss)[range(batch_obs.shape[0]), list(batch_a)]

        loss = self.criterion(q_pred, y)
        #print (loss.data[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_to_memory(self, obs, action, reward, done, next_obs):
        s = Tensor(obs).view(-1, self.s_dim)
        s1 = Tensor(next_obs).view(-1, self.s_dim)
        self.memory.add_sample(s, action, reward, done*1.0, s1)

    def clear_memory(self):
        self.memory.clear()
# -------------------------------------------------------------------

# -------------------- DQN class ------------------------------------

#'''
class DQN(nn.Module):
    def __init__(self, s_dim, a_dim, gamma):
        super(DQN, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = 256
        self.h2_dim = 256
        self.h3_dim = 64

        self.l1 = nn.Linear(self.s_dim, self.h1_dim)
        self.l2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.l3 = nn.Linear(self.h2_dim, self.h3_dim)
        self.l4 = nn.Linear(self.h3_dim, self.a_dim)

        self.memory = Buffer()
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        y = self.l4(y)
        return y

    # TODO: supports only one input vector for now (not sure though)
    def policy(self, x, eps=.05):
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.a_dim)
        else:
            obs = torch.from_numpy(x).view(-1, self.s_dim)
            inp = Variable(obs, volatile = True).type(FloatTensor)
            Qvals = self.forward(inp).data
            _, a = torch.max(Qvals, 1)
            return a[0]

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
        Q1, _ = torch.max(self.forward(inp), dim=1)
        y = batch_r + self.gamma * (1-batch_done) * Q1

        ss = Variable(batch_obs).type(FloatTensor)
        Q_pred = self.forward(ss)[range(batch_size), list(batch_a)]

        loss = self.criterion(Q_pred, y)
        #print (loss.data[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_to_memory(self, obs, action, reward, done, next_obs):
        s = Tensor(obs).view(-1, self.s_dim)
        s1 = Tensor(next_obs).view(-1, self.s_dim)
        self.memory.add_sample(s, action, reward, done*1.0, s1)
#'''

'''
class DQN(nn.Module):
    def __init__(self, s_dim, a_dim, gamma):
        super(DQN, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = 256
        self.h2_dim = 64

        self.l1 = nn.Linear(self.s_dim, self.h1_dim)
        self.l2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.l3 = nn.Linear(self.h2_dim, self.a_dim)

        self.memory = Buffer()
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = self.l3(y)
        return y

    # TODO: supports only one input vector for now (not sure though)
    def policy(self, x, eps=.05):
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.a_dim)
        else:
            obs = torch.from_numpy(x).view(-1, self.s_dim)
            inp = Variable(obs, volatile = True).type(FloatTensor)
            Qvals = self.forward(inp).data
            _, a = torch.max(Qvals, 1)
            return a[0]

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
        Q1, _ = torch.max(self.forward(inp), dim=1)
        y = batch_r + self.gamma * (1-batch_done) * Q1

        ss = Variable(batch_obs).type(FloatTensor)
        Q_pred = self.forward(ss)[range(batch_size), list(batch_a)]

        loss = self.criterion(Q_pred, y)
        #print (loss.data[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_to_memory(self, obs, action, reward, done, next_obs):
        s = Tensor(obs).view(-1, self.s_dim)
        s1 = Tensor(next_obs).view(-1, self.s_dim)
        self.memory.add_sample(s, action, reward, done*1.0, s1)
'''
# ----------------------------------------------------------------------

# ------------------ Dueling DQN --------------------------------------

# stands for Dueling DQN (and not Double DQN)
class DDQN(nn.Module):
    def __init__(self, s_dim, a_dim, gamma):
        super(DDQN, self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = 256
        self.h2_dim = 256
        self.h3_dim = 64

        self.l1 = nn.Linear(self.s_dim, self.h1_dim)
        self.l2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.l3 = nn.Linear(self.h2_dim, self.h3_dim)
        self.advantage = nn.Linear(self.h3_dim, self.a_dim)
        self.value = nn.Linear(self.h3_dim, 1)

        self.memory = Buffer()
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        adv = self.advantage(y)
        val = self.value(y)
        y = adv + val
        return y

    # TODO: supports only one input vector for now (not sure though)
    def policy(self, x, eps=.05):
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.a_dim)
        else:
            obs = torch.from_numpy(x).view(-1, self.s_dim)
            inp = Variable(obs, volatile = True).type(FloatTensor)
            Qvals = self.forward(inp).data
            _, a = torch.max(Qvals, 1)
            return a[0]

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
        Q1, _ = torch.max(self.forward(inp), dim=1)
        y = batch_r + self.gamma * (1-batch_done) * Q1

        ss = Variable(batch_obs).type(FloatTensor)
        Q_pred = self.forward(ss)[range(batch_size), list(batch_a)]

        loss = self.criterion(Q_pred, y)
        #print (loss.data[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def add_to_memory(self, obs, action, reward, done, next_obs):
        s = Tensor(obs).view(-1, self.s_dim)
        s1 = Tensor(next_obs).view(-1, self.s_dim)
        self.memory.add_sample(s, action, reward, done*1.0, s1)
# ----------------------------------------------------------------------


# note - 2 hidden layer deep networks works fine, no need for 3
# set hidden layer size to be (256,64) - works good

if __name__ == '__main__':
    # gamma depends on the environment, so this setup
    env_names = ['MountainCar-v0', 'CartPole-v0']
    gammas = [1, .99]

    # 0 - mountaincar || 1 - cartpole
    env_id = 0
    gamma = gammas[env_id]
    env_name = env_names[env_id]

    env = gym.make(env_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    #mdl = QN_linear(s_dim, a_dim, gamma)
    mdl = DQN(s_dim, a_dim, gamma)
    #mdl = DDQN(s_dim, a_dim, gamma)

    if use_cuda:
        mdl = mdl.cuda()

    nepisodes = 10000
    episode_max_length = 1000
    batch_size = 256
    update_frequency = 4

    e = 1.0
    e_min = .05
    annealing_steps = int(1e6)
    e_step = (e - e_min)/annealing_steps

    i = 0
    steps = 0
    render = False
    rall = []

    while i < nepisodes:
        i += 1
        s = env.reset()

        j = 0
        episode_reward = 0
        while j < episode_max_length:
            j += 1
            steps += 1
            action = mdl.policy(s, eps=e)
            e = max(e_min, e - e_step)
            s1, r, done, _ = env.step(action)
            mdl.add_to_memory(s, action, r, done, s1)

            if render:
                env.render()

            s = s1[:]
            episode_reward += r

            if steps > batch_size and steps % update_frequency == 0:
                mdl.train(batch_size)

            if done:
                #mdl.train()
                #mdl.clear_memory()
                break

        rall.append(episode_reward)
        print ('Episode: %3d, e: %.4f, Reward: %3d, Avg Reward: %.2f'
               %(i, e, episode_reward, np.mean(rall[-50:])))

