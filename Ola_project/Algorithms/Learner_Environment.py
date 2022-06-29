import numpy as np
from Environment.E_commerce import *

class Environment:
  #take as input the E-commerce object with the prices already changed
  #and the day_index to access to the informations
    def __init__(self, n_arms, E_commerce, product_index, num_users, binary_vector, fixed_alpha):
        self.n_arms = n_arms
        self.E = E_commerce
        self.day = 0
        self.product_idx = product_index
        self.clicks_current_day = 0  #each day (round) they reset
        self.purchases_current_day  = 0 #each day (round) they reset
        self.num_users = num_users
        self.binary_vector = binary_vector
        self.fixed_alpha = fixed_alpha
        self.margins_list = E_commerce.products[product_index].margins_list

    def round(self, pulled_arm):
      #Reward is given thanks to the simulation of a day in the E-commerce website
        self.E.products[self.product_idx].change_price(pulled_arm)
        self.E.simulate_day(self.num_users, self.binary_vector, self.fixed_alpha)
        self.clicks_current_day = self.E.daily_clicks[self.product_idx]
        self.purchases_current_day = self.E.daily_purchases[self.product_idx]
        conv_rate = self.purchases_current_day/self.clicks_current_day
        reward = self.E.products[self.product_idx].margin * conv_rate
        self.day += 1
        return reward


class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[0] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
