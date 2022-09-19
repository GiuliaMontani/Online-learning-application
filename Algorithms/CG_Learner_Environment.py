import numpy as np
from Environment.CG_E_commerce import *
import copy


class CG_Environment:

    def __init__(self, n_arms, E_commerce, margins_matrix, num_users, fixed_alpha, fixed_weights,
                 fixed_units):
        self.n_arms = n_arms
        self.E = copy.deepcopy(E_commerce)
        self.clicks_current_day = [[0 for _ in range(2)] for _ in range(2)] # each day (round) they reset
        self.purchases_current_day = [[0 for _ in range(2)] for _ in range(2)]  # each day (round) they reset
        self.num_users = num_users
        self.fixed_alpha = fixed_alpha  # flag for the alpha ratios uncertainity
        self.margins_matrix = margins_matrix
        self.fixed_weights = fixed_weights
        self.daily_units = [[np.zeros(5) for _ in range(2)] for _ in range(2)]
        self.fixed_units = fixed_units

    def round(self, pulled_arm):
        """For each product, it changes the price corresponding to the pulled arm

        :param pulled_arm: vector with pulled prices
        :type pulled_arm: ndarray
        :return: reward
        """
        for f1 in range(2):
            for f2 in range(2):
                for i in range(5):
                    self.E.products[f1][f2][i].change_price(int(pulled_arm[f1][f2][i]))

        # Reward is given thanks to the simulation of a day in the E-commerce website
        self.E.simulate_day(self.num_users, self.fixed_alpha, self.fixed_weights, self.fixed_units)
        self.clicks_current_day = self.E.daily_clicks
        self.purchases_current_day = self.E.daily_purchases
        self.daily_units = self.E.daily_purchased_units

        # reward for each product is the sum of rewards per product / clicks on that product
        reward = [[np.zeros(5) for _ in range(2)] for _ in range(2)]
        for f1 in range(2):
            for f2 in range(2):
                for i in range(5):
                    if self.clicks_current_day[f1][f2][i] != 0:
                        reward[f1][f2][i] = self.E.daily_rewards_per_product[f1][f2][i] / self.clicks_current_day[f1][f2][i]
        return reward