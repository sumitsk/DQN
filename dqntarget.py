#!/usr/bin/env python

# ------------- import everything you need -------------
import gym
import numpy as np
import random, os, sys, cv2
from collections import namedtuple
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle

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


def save_checkpoint(state, filename):
    torch.save(state, filename)


def preprocess_image(obs):
    gray = np.dot(obs[..., :3], [.299, .587, .114])
    img = cv2.resize(gray, (84, 84))
    img = img / 255.0
    return img

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

    # length of data buffer
    def __len__(self):
        return len(self.data)
# ------------------------------------------------------------------


class QN_linear(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(QN_linear, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.l1 = nn.Linear(self.s_dim, self.a_dim)

    def forward(self, x):
        y = self.l1(x)
        return y
# -------------------------------------------------------------------


class DQN(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(DQN, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = 32
        self.h2_dim = 64
        self.h3_dim = 32

        self.l1 = nn.Linear(self.s_dim, self.h1_dim)
        self.l2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.l3 = nn.Linear(self.h2_dim, self.h3_dim)
        self.l4 = nn.Linear(self.h3_dim, self.a_dim)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        y = self.l4(y)
        return y
# ----------------------------------------------------------------------

# ------------------ Dueling DQN --------------------------------------

# stands for Dueling DQN (and not Double DQN)


class DDQN(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(DDQN, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_dim = 32
        self.h2_dim = 64
        self.h3_dim = 32

        self.l1 = nn.Linear(self.s_dim, self.h1_dim)
        self.l2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.l3 = nn.Linear(self.h2_dim, self.h3_dim)
        self.advantage = nn.Linear(self.h3_dim, self.a_dim)
        self.value = nn.Linear(self.h3_dim, 1)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        y = F.relu(self.l3(y))
        adv = self.advantage(y)
        val = self.value(y)
        y = adv + val
        return y
# ----------------------------------------------------------------------


# -------------------------  Atari DQN -------------------------------------
class DQN_atari(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(DQN_atari, self).__init__()
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

# ------------------ Atari Dueling DQN --------------------------------------


class DDQN_atari(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(DDQN_atari, self).__init__()
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

# -----------------------------------------------------------------------------------------


# -------------------- DQN class ------------------------------------


class DQNmodel(object):
    def __init__(self, env, gamma, network='dqn'):
        self.env = deepcopy(env)
        self.network = network
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]

        print (network)

        if network == 'dqn':
            self.main = DQN(self.s_dim, self.a_dim)
            self.target = DQN(self.s_dim, self.a_dim)
        elif network == 'ddqn':
            self.main = DDQN(self.s_dim, self.a_dim)
            self.target = DDQN(self.s_dim, self.a_dim)
        elif network == 'linear_qn' or network == 'linear_qn_replay':
            self.main = QN_linear(self.s_dim, self.a_dim)
            self.target = QN_linear(self.s_dim, self.a_dim)
        elif network == 'dqn_atari':
            self.main = DQN_atari(self.s_dim, self.a_dim)
            self.target = DQN_atari(self.s_dim, self.a_dim)
        elif network == 'ddqn_atari':
            self.main = DDQN_atari(self.s_dim, self.a_dim)
            self.target = DDQN_atari(self.s_dim, self.a_dim)

        else:
            message = 'Unknown network!! \n Please enter one of the following:\n dqn\n ddqn\n ' \
                      'linear_qn \n linear_qn_replay\n Exiting the program now'
            sys.exit(message)

        if use_cuda:
            self.main.cuda()
            self.target.cuda()

        self.atari = True if network == 'dqn_atari' or network == 'ddqn_atari' else False
        buffer_size = 125000 if self.atari else int(1e6)
        lr = 2.5*1e-4 if self.atari else 1e-3
        tau = .001

        self.test_rewards = []
        self.test_episodes = []
        self.train_episodes = []
        self.train_rewards = []

        self.memory = Buffer(max_length=buffer_size)
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.main.parameters(), lr=lr)
        self.tau = tau
        hard_update(self.target, self.main)             # set parameters to be equal at the beginning

    def train(self, batch_size):
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
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.target, self.main, self.tau)

    def test(self, episode_num, nepisodes=20, return_average=True):
        total_reward = 0
        all_episode_rewards = []
        eps = .05
        for _ in range(nepisodes):
            episode_reward = 0
            s = self.env.reset()
            for _ in range(episode_max_length):
                action = self.policy(s, eps=eps)
                s1, r, done, _ = self.env.step(action)
                episode_reward += r
                s = s1[:]
                if done:
                    break
            all_episode_rewards.append(episode_reward)
            total_reward += episode_reward

        if return_average:
            average_reward = total_reward/nepisodes
            self.test_episodes.append(episode_num)
            self.test_rewards.append(average_reward)
            return average_reward

        return all_episode_rewards

    def test_atari(self, episode_num, nepisodes=5, return_average=True):
        total_reward = 0
        eps = 0.05
        stack_size = 4
        all_episode_rewards = []

        for _ in range(nepisodes):
            image_stack = []
            s = self.env.reset()
            s = preprocess_image(s)
            for _ in range(stack_size):
                image_stack.append(s)

            episode_reward = 0
            done = False
            while not done:
                action = mdl.policy(image_stack, eps=eps)
                s1, r, done, _ = self.env.step(action)
                s1 = preprocess_image(s1)
                image_stack.append(s1)
                image_stack.pop(0)
                episode_reward += r

            all_episode_rewards.append(episode_reward)
            total_reward += episode_reward

        if return_average:
            average_reward = total_reward / nepisodes
            self.test_episodes.append(episode_num)
            self.test_rewards.append(average_reward)
            return average_reward
        return all_episode_rewards

    def add_to_memory(self, obs, action, reward, done, next_obs):
        s = torch.unsqueeze(torch.from_numpy(np.array(obs)), 0)
        s1 = torch.unsqueeze(torch.from_numpy(np.array(next_obs)), 0)
        self.memory.add_sample(s, action, reward, done*1.0, s1)

    def policy(self, x, eps=.05):
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.a_dim)
        else:
            obs = torch.unsqueeze(torch.from_numpy(np.array(x)), 0)
            inp = Variable(obs, volatile=True).type(FloatTensor)
            q_values = self.main.forward(inp).data
            _, a = torch.max(q_values, 1)
            return a[0]

    def save(self, episode_num, filename):
        save_checkpoint({
            'episode_number': episode_num,
            'main_state_dict': self.main.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename=filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.main.load_state_dict(checkpoint['main_state_dict'])
        self.target.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        episode_number = checkpoint['episode_num']
        return episode_number

    def add_train_reward(self, episode_num, reward):
        self.train_rewards.append(reward)
        self.train_episodes.append(episode_num)

    def plot_rewards(self):
        plt.figure()
        plt.scatter(self.test_episodes, self.test_rewards, s=30, label="Average test reward", marker='x', c='b')
        plt.scatter(self.train_episodes, self.train_rewards, s=2, label="Training reward", marker='o', c='r')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend()

        # due to some reason, the after else statement returns none error for space invaders env
        env_name = 'SpaceInvaders' if self.atari else self.env.env._spec._env_name
        plt.title(env_name)
        fn = env_name + '_' + network
        plt.savefig(fn + '.jpg')
        #plt.show()

        with open(fn + '.pickle', "wb") as output_file:
            dictionary = {'training_episodes': self.train_episodes,
                          'test_episodes': self.test_episodes,
                          'training_rewards': self.train_rewards,
                          'test_rewards': self.test_rewards}
            pickle.dump(dictionary, output_file)

    def final_test_rewards(self, num_episodes=2):
        all_rewards = self.test(0, nepisodes=num_episodes, return_average=False)
        mu = np.mean(all_rewards)
        std = np.std(all_rewards)
        return mu, std

    def final_test_rewards_atari(self, num_episodes=2):
        all_rewards = self.test_atari(0, nepisodes=num_episodes, return_average=False)
        mu = np.mean(all_rewards)
        std = np.std(all_rewards)
        return mu, std


if __name__ == '__main__':
    # gamma depends on the environment, so this setup
    env_names = ['MountainCar-v0', 'CartPole-v0', 'SpaceInvaders-v0']
    gammas = [1, .99, .99]
    env_id = 2                  # change this to setup different gym environments
    gamma = gammas[env_id]
    env_name = env_names[env_id]

    env = gym.make(env_name)
    #env = gym.wrappers.Monitor(env, './videos', force=True, video_callable=lambda ep_id: ep_id % 100 == 0)

    networks = ['dqn', 'ddqn', 'linear_qn', 'linear_qn_replay']
    network_id = 0              # change this to setup different architecture of the model
    network = networks[network_id]

    if env_name is 'SpaceInvaders-v0':
        network = network + '_atari'
    mdl = DQNmodel(env, gamma, network=network)

    # '''
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    # '''

    if env_name == 'SpaceInvaders-v0':
        nepisodes = 2
        episode_max_length = env._max_episode_steps + 1
        batch_size = 32
        update_frequency = 4             # after .... iterations
        save_freq = nepisodes/10         # after .... episodes
        test_frequency = 100             # after .... episodes

        e = 1.0
        e_min = .05
        annealing_steps = int(1e6)  # 1 million frames
        e_step = (e - e_min) / annealing_steps

        i = 0
        steps = 0
        render = False
        action_freq = 3
        stack_size = 4
        image_stack = []
        pre_train_steps = batch_size + stack_size

        directory = 'models/' + env_name + '/' + network + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        while i < nepisodes:
            i += 1
            s = env.reset()
            s = preprocess_image(s)

            if i % test_frequency == 0:
                test_reward = mdl.test_atari(i)

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
                # print (i, j, r, done)
                s1 = preprocess_image(s1)

                next_image_stack = image_stack[:]
                next_image_stack.append(s1)
                next_image_stack.pop(0)

                rwd = r / abs(r) if r != 0 else 0
                mdl.add_to_memory(image_stack, action, rwd, done, next_image_stack)
                image_stack = next_image_stack[:]

                if render:
                    env.render()

                s = s1[:]
                episode_reward += r

                if steps > pre_train_steps and steps % update_frequency == 0:
                    mdl.train(batch_size)

                if done:
                    break

            mdl.add_train_reward(i, episode_reward)
            print ('Episode: %5d, e: %.4f, Reward: %4d, Avg Reward: %.2f'
                   % (i, e, episode_reward, np.mean(mdl.train_rewards[-100:])))

            if i == 1 or i % save_freq == 0:
                filename = directory + '/episode_' + str(i) + '.pt'
                mdl.save(i, filename)

        print ('Generating reward plots')
        mdl.plot_rewards()
        print ('Reward plots saved')
        mu, std = mdl.final_test_rewards_atari()
        print ('Trained model statistics: \n mean: %.4f\n std: %.4f' % (mu, std))

    else:
        nepisodes = 7000
        episode_max_length = env._max_episode_steps + 1
        batch_size = 32
        update_frequency = 1            # after .... iterations
        save_freq = nepisodes/5         # after .... episodes
        test_frequency = 100            # after .... episodes

        e = 1.0
        e_min = .05
        annealing_steps = int(1e5)
        e_step = (e - e_min)/annealing_steps

        i = 0
        steps = 0
        render = False

        directory = 'models/' + env_name + '/' + network + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        while i < nepisodes:
            i += 1
            s = env.reset()

            if i % test_frequency == 0:
                test_reward = mdl.test(i)

            j = 0
            episode_reward = 0
            while j < episode_max_length:
                j += 1
                steps += 1
                action = mdl.policy(s, eps=e)
                e = max(e_min, e - e_step)
                s1, r, done, _ = env.step(action)

                # modifying done state for linear network in mountain car environment
                if env_id == 0 and (network == 'linear_qn' or network == 'linear_qn_replay'):
                    done_state = 1.0 if s1[0] > .5 else 0.0
                else:
                    done_state = done
                mdl.add_to_memory(s, action, r, done_state, s1)

                if render:
                    env.render()

                s = s1[:]
                episode_reward += r

                if steps > batch_size and steps % update_frequency == 0:
                    mdl.train(batch_size)

                if done:
                    break

            mdl.add_train_reward(i, episode_reward)
            print ('Episode: %3d, e: %.4f, Reward: %3d, Avg Reward: %.2f'
                   % (i, e, episode_reward, np.mean(mdl.train_rewards[-100:])))

            if i == 1 or i % save_freq == 0:
                filename = directory + '/episode_' + str(i) + '.pt'
                mdl.save(i, filename)

        print ('Generating reward plots')
        mdl.plot_rewards()
        print ('Reward plots saved')
        mu, std = mdl.final_test_rewards()
        print ('Trained model statistics: \n mean: %.4f\n std: %.4f' % (mu, std))

