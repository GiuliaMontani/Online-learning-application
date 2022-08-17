from Algorithms.Learner_Environment import *


class TS_poisson(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.array([np.ones((n_arms, 2))] * 5)
        self.expected_rewards = np.zeros([5, n_arms])
        self.lambda_poisson = np.array([np.zeros(4)] * 5)  # mean quantities bought per product
        # (in step 4 they are variable)

    def pull_arm(self, margins_matrix):
        # First exploration of all arms (to get a first estimate of their conversion rate)
        if self.t < self.n_arms:
            return np.array([self.t] * 5)

        idx = np.zeros(5)
        # We pull the arm with the best expected reward drawn from a beta with respect to the margins (lower conv
        # rates have higher margins) and we consider the units bought distributed as a poisson with lambda = mean
        # units bought per arm until now
        for i in range(5):
            idx[i] = np.argmax(
                np.random.beta(self.beta_parameters[i][:, 0], self.beta_parameters[i][:, 1]) * margins_matrix[i, :]
                * self.lambda_poisson[i])  # * np.random.poisson(self.lambda_poisson[i], size = 4))
        return idx

    def update(self, pulled_arm, reward, clicks, purchases, daily_units):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        for i in range(5):
            # we update the beta with the clicks and purchases of the day: estimation of the conversion rates
            self.beta_parameters[i][int(pulled_arm[i]), 0] = self.beta_parameters[i][int(pulled_arm[i]), 0] + purchases[
                i]
            self.beta_parameters[i][int(pulled_arm[i]), 1] = self.beta_parameters[i][int(pulled_arm[i]), 1] + (
                        clicks[i] - purchases[i])
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

            # dobbiamo inserire qui il calcolo di alpha?


