from Algorithms.Learner_Environment import *
# from Algorithms.TS_Learner_poisson import *+
from Algorithms.UCB_Learner import UCB
import numpy as np


class SW_UCB(UCB):
    def __init__(self, n_arms, window_size):
        """ Sliding-window UCB algorithm.

        :param n_arms: number of prices
        :param window_size: size of the window (constant)
        """
        super().__init__(n_arms)
        self.rewards_per_arm = [[[] for i in range(n_arms)] for j in range(5)]
        '''
        self.expected_rewards = np.zeros([5, n_arms])
        self.confidence = np.array(([[np.inf] * n_arms] * 5))
        self.make_comparable = np.zeros(5)
        self.explore = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        self.c = 2
        '''
        self.window_size = window_size
        self.pulled_arms = [np.array([]) for j in range(5)]

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        print('-----------time ', self.t)
        print('pulled ',pulled_arm)
        for i in range(5):
            # for each product i
            self.pulled_arms[i] = np.append(self.pulled_arms[i], pulled_arm[i])
            self.counter_per_arm[i][int(pulled_arm[i])] += 1

            #print('time t:',self.t)
            #print('expected rewards', self.expected_rewards)
            #print('product ',i)
            for arm in range(self.n_arms):
                n_samples = np.sum(self.pulled_arms[i][-self.window_size:-1] == arm)
                #print('arm', arm, ', n', n_samples)
                #print('rewards till now: ',self.rewards_per_arm[i][arm])
                self.expected_rewards[i][arm] = np.mean(self.rewards_per_arm[i][arm][-n_samples:])
                #print('expected rewards', self.expected_rewards)

                if self.t < self.window_size:
                    self.confidence[i][arm] = (self.c * np.log(
                        self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
                else:
                    self.confidence[i][arm] = (self.c * np.log(
                        self.window_size) / n_samples) ** 0.5 if n_samples > 0 else np.inf


