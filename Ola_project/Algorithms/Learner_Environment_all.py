import numpy as np
from Environment.E_commerce import *

class Environment_all:
  #take as input the E-commerce object with the prices already changed
  #and the day_index to access to the informations
    def __init__(self, n_arms, E_commerce, margins_matrix, num_users, binary_vector, fixed_alpha):
        self.n_arms = n_arms
        self.E = E_commerce
        self.day = 0
        self.clicks_current_day = 0  #each day (round) they reset
        self.purchases_current_day  = 0 #each day (round) they reset
        self.num_users = num_users
        self.binary_vector = binary_vector
        self.fixed_alpha = fixed_alpha
        self.margins_matrix = margins_matrix

    def round(self, pulled_arm):
      #Reward is given thanks to the simulation of a day in the E-commerce website
        for i in range(5):
            self.E.products[i].change_price(int(pulled_arm[i]))
        
        self.E.simulate_day(self.num_users, self.binary_vector, self.fixed_alpha)
        self.clicks_current_day = self.E.daily_clicks
        self.purchases_current_day = self.E.daily_purchases
        conv_rate = self.purchases_current_day/self.clicks_current_day

        reward = np.zeros(5)
        for i in range(5):
            reward[i] = self.margins_matrix[i,int(pulled_arm[i])] * conv_rate[i]
        self.day += 1
        return reward


class Learner_all:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[[0] for i in range(n_arms) ]for j in range(5)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        for i in range(5):
            self.rewards_per_arm[i][int(pulled_arm[i])].append(reward[i])
        self.collected_rewards = np.append(self.collected_rewards, np.sum(reward))
