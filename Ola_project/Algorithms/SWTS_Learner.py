from Algorithms.TS_Learner import TS
# from Algorithms.TS_Learner_poisson import *
import numpy as np


class SW_TS(TS):
    def __init__(self, n_arms, window_size):
        """ Sliding-window Thompson Sampling algorithm.

        :param n_arms: number of prices
        :param window_size: size of the window (constant)
        """
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = [np.array([]) for j in range(5)]
        self.clicks = [[[] for i in range(n_arms)] for j in range(5)]
        self.purchases = [[[] for i in range(n_arms)] for j in range(5)]
        self.n_bought_products = [[[] for i in range(n_arms)] for j in range(5)]

    '''
    def update_observations(self, pulled_arm, reward):
        for i in range(5):
            self.rewards_per_arm[i][int(pulled_arm[i])].append(reward[i])
        self.collected_rewards = np.append(self.collected_rewards, np.sum(reward))
    '''

    def pull_arm(self, margins_matrix):
        # First exploration of all arms (to get a first estimate of their conversion rate)
        if self.t < self.n_arms:
            return np.array([self.t] * 5)
        idx = np.zeros(5)
        # We pull the arm with the best expected reward drawn from a beta with respect to the margins
        # (lower conv rates have higher margins)
        for i in range(5):
            '''
            print(self.beta_parameters[i][:, 0])
            print(self.beta_parameters[i][:, 0] <= 0)
            print(np.any(self.beta_parameters[i][:, 0] <= 0))
            print(np.any(self.beta_parameters[i][:, 1] <= 0))
            '''

            if np.any(self.beta_parameters[i][:, 0] <= 0) or np.any(self.beta_parameters[i][:, 1] <= 0):
                print('cond1',np.any(self.beta_parameters[i][:, 0] <= 0))
                print('cond2',np.any(self.beta_parameters[i][:, 1] <= 0))
                print('beta1',self.beta_parameters[i][:, 0])
                self.beta_parameters[i][:, 0] += 1
                print('beta1-b',self.beta_parameters[i][:, 0])
                print('beta2',self.beta_parameters[i][:, 1])
                self.beta_parameters[i][:, 1] += 1
                print('beta2-b',self.beta_parameters[i][:, 1])
                idx[i] = int(np.argmax(
                    np.random.beta(self.beta_parameters[i][:, 0]+1,
                                   self.beta_parameters[i][:, 1]+1) * margins_matrix[i, :]))
                print(idx[i])
            else:
                idx[i] = np.argmax(
                    np.random.beta(self.beta_parameters[i][:, 0], self.beta_parameters[i][:, 1]) * margins_matrix[i, :])

    def update(self, pulled_arm, reward, click, purchase, daily_units):
        self.t += 1
        # print(self.t)
        # print(pulled_arm)
        self.update_observations(pulled_arm, reward)

        # print(click)
        # print(purchase)

        for i in range(5):
            # for each product i
            self.pulled_arms[i] = np.append(self.pulled_arms[i], pulled_arm[i])

            for arm in range(self.n_arms):
                # update arm
                if arm == int(pulled_arm[i]):
                    self.clicks[i][arm].append(click[i])
                    self.purchases[i][arm].append(purchase[i])
                    self.n_bought_products[i][arm].append(daily_units[i])
                #else:
                    #self.clicks[i][arm].append(0)
                    #self.purchases[i][arm].append(0)
                    #self.n_bought_products[i][arm].append(0)

        # print('pulled arms:', self.pulled_arms)
        for i in range(5):
            # for each product i
            for arm in range(self.n_arms):
                # update parameters considering only the window_size
                n_samples = np.sum(self.pulled_arms[i][-self.window_size:] == arm)
                # print('arm', arm, ': ', n_samples)
                cum_purchases = np.sum(self.purchases[i][arm][-n_samples:]) if n_samples > 0 else 0
                # print(n_samples > 0)
                # print(self.purchases[i][arm])
                # print(self.purchases[i][arm][-n_samples:])
                # print('cum_purchases',cum_purchases)
                cum_clicks = np.sum(self.clicks[i][arm][-n_samples:]) if n_samples > 0 else 0
                n_bought_products = np.sum(
                    self.n_bought_products[i][arm][-n_samples:]) if n_samples > 0 else 0

                self.beta_parameters[i][arm, 0] = cum_purchases if n_samples > 0 else 1
                self.beta_parameters[i][arm, 1] = cum_clicks - cum_purchases if n_samples > 0 else 1

            print(self.beta_parameters)

        '''
                    self.counter_per_arm[i][int(pulled_arm[i])] += 1
                    self.expected_rewards[i][int(pulled_arm[i])] = (self.expected_rewards[i][int(pulled_arm[i])] * (
                            self.counter_per_arm[i][int(pulled_arm[i])] - 1) + reward[i]) / self.counter_per_arm[i][
                                                                       int(pulled_arm[i])]


                    # update the lambda only if bought
                    if purchases[i] != 0:
                        self.lambda_poisson[i][int(pulled_arm[i])] = (self.lambda_poisson[i][int(pulled_arm[i])] * (
                                self.counter_per_arm[i][int(pulled_arm[i])] - 1) + daily_units[i] / purchases[i]) / \
                                                                     self.counter_per_arm[i][int(pulled_arm[i])] if \
                            self.lambda_poisson[i][int(pulled_arm[i])] > 0 else daily_units[i] / purchases[i]

        '''
