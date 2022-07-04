import numpy as np
from Environment.E_commerce import *

class Environment:

    def __init__(self, n_arms, E_commerce, margins_matrix, num_users, binary_vector, fixed_alpha, fixed_weights):
        self.n_arms = n_arms
        self.E = E_commerce
        self.clicks_current_day = 0  #each day (round) they reset
        self.purchases_current_day  = 0 #each day (round) they reset
        self.num_users = num_users
        self.binary_vector = binary_vector #defines the class of users
        self.fixed_alpha = fixed_alpha #flag for the alpha ratios uncertainity
        self.margins_matrix = margins_matrix
        self.fixed_weights = fixed_weights

    def round(self, pulled_arm):

        #for each product we change the price corresponding to the arm pulled
        for i in range(5):
            self.E.products[i].change_price(int(pulled_arm[i]))

        #Reward is given thanks to the simulation of a day in the E-commerce website
        self.E.simulate_day(self.num_users, self.binary_vector, self.fixed_alpha, self.fixed_weights)
        self.clicks_current_day = self.E.daily_clicks
        self.purchases_current_day = self.E.daily_purchases
        #computation of the conv_rates of the day for the arm pulled (for each product)
        self.clicks_current_day[np.argwhere(self.clicks_current_day==0)] = 1 #to avoid /0
        conv_rate = self.purchases_current_day/self.clicks_current_day

        #the expected reward is given by the product between the margin and the conv rate of the arm pulled (for each product)
        #we estimate the expected reward and not the total reward: we don't really care about the number of users
        #but more users in the day can get a better estimation of the conv rates
        reward = np.zeros(5)
        for i in range(5):
            reward[i] = self.margins_matrix[i,int(pulled_arm[i])] * conv_rate[i]
        return reward


class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[[0] for i in range(n_arms) ]for j in range(5)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        for i in range(5):
            self.rewards_per_arm[i][int(pulled_arm[i])].append(reward[i])
        self.collected_rewards = np.append(self.collected_rewards, np.sum(reward))
