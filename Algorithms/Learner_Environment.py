import numpy as np
from Environment.E_commerce import *
import copy

def reset_users_reserv_prices():
    #User0.reset_avg_reservation_price()
    #User1.reset_avg_reservation_price()
    #User2.reset_avg_reservation_price()
    User0.avg_reservation_price = np.array([23, 34, 31, 46, 104])
    User1.avg_reservation_price = np.array([21, 32, 29, 44, 95])
    User2.avg_reservation_price = np.array([21, 31, 28, 42, 87])


class Environment:

    def __init__(self, n_arms, E_commerce, margins_matrix, num_users, fixed_alpha, fixed_weights,
                 fixed_units):
        self.n_arms = n_arms
        self.E = copy.deepcopy(E_commerce)
        self.clicks_current_day = 0  # each day (round) they reset
        self.purchases_current_day = 0  # each day (round) they reset
        self.num_users = num_users
        self.fixed_alpha = fixed_alpha  # flag for the alpha ratios uncertainity
        self.margins_matrix = margins_matrix
        self.fixed_weights = fixed_weights
        self.daily_units = np.zeros(5)
        self.fixed_units = fixed_units
        reset_users_reserv_prices()  # for abrupt change

    def round(self, pulled_arm):
        """For each product, it changes the price corresponding to the pulled arm

        :param pulled_arm: vector with pulled prices
        :type pulled_arm: ndarray
        :return: reward
        """

        for i in range(5):
            self.E.products[i].change_price(int(pulled_arm[i]))

        # Reward is given thanks to the simulation of a day in the E-commerce website
        self.E.simulate_day(self.num_users, self.fixed_alpha, self.fixed_weights, self.fixed_units)
        self.clicks_current_day = self.E.daily_clicks
        self.purchases_current_day = self.E.daily_purchases
        self.daily_units = self.E.daily_purchased_units

        # reward for each product is the sum of rewards per product / clicks on that product
        reward = np.zeros(5)
        for i in range(5):
            if self.clicks_current_day[i] != 0:
                reward[i] = self.E.daily_rewards_per_product[i] / self.clicks_current_day[i]
        return reward

    def abrupt_change(self, changing_users, percentage):
        """For each user class, it changes the reservation prices by a percentage
                :param changing_users: which user changes
                :type percentage: percentage of change
        """
        print('--------------------------------')
        print('--------------------------------')
        print("Abrupt change")
        print('--------------------------------')
        print('--------------------------------')
        for user in changing_users:
            if user == 0:
                User0.avg_reservation_price = np.multiply(percentage[0],User0.avg_reservation_price)
                print('class User 0 has changed: its new average reservation price is', User0.avg_reservation_price)
            if user == 1:
                User1.avg_reservation_price = np.multiply(percentage[1],User1.avg_reservation_price)
                print('class User 1 has changed: its new average reservation price is', User1.avg_reservation_price)
            if user == 2:
                User2.avg_reservation_price = np.multiply(percentage[2],User2.avg_reservation_price)
                print('class User 2 has changed: its new average reservation price is', User2.avg_reservation_price)


class Learner:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[[] for i in range(n_arms)] for j in range(5)]
        self.collected_rewards = np.array([])
        self.counter_per_arm = np.array([np.zeros(4)] * 5)

    def update_observations(self, pulled_arm, reward):
        for i in range(5):
            self.rewards_per_arm[i][int(pulled_arm[i])].append(reward[i])
        self.collected_rewards = np.append(self.collected_rewards, np.sum(reward))
        
    def reset(self):
        self.rewards_per_arm = [[[] for i in range(self.n_arms)] for j in range(5)]  #perchè 0?
        self.collected_rewards = np.array([])
        self.counter_per_arm = np.array([np.zeros(4)] * 5)



class LinearMabEnvironment(Environment):

    # dim = dimension of the feature vector ( = 2)
    def __init__(self, n_arms, E_commerce, margins_matrix, num_users, fixed_alpha, fixed_weights,
                 fixed_units, dim):
        super().__init__(n_arms, E_commerce, margins_matrix, num_users, fixed_alpha, fixed_weights,
                 fixed_units)
        self.theta = np.random.dirichlet(np.ones(dim), size = 1) # set the values of the parameters such that they sum to 1
        self.arms_features = np.random.binomial(1, 0.5, size=(n_arms,dim))
        self.p = np.zeros(n_arms)
        for i in range(0, n_arms):
            self.p[i] = np.dot(self.theta, self.arms_features[i])



