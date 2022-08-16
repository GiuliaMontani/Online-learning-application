from Algorithms.Learner_Environment import *


class Conv_rates(Learner):
    def __init__(self, n_arms):
        """Greedy Algorithm (with random choice)
        This simple algorithm will be used to obtain the optimal reward after repeated long simulations.

        :param n_arms: number of possible prices for each product (4 in our case)
        :type n_arms: int
        """
        super().__init__(n_arms)
        self.beta_parameters = np.array([np.ones((n_arms, 2))] * 5)
        self.expected_rewards = np.zeros([5, n_arms])
        self.lambda_poisson = np.array([np.zeros(4)] * 5)

    def pull_arm(self, best_arm_per_product=[], clairvoyant_flag=0):
        if clairvoyant_flag == 1:
            return best_arm_per_product

        return np.random.choice(4, size=5)

    def update(self, pulled_arm, reward, clicks, purchases, daily_units):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        for i in range(5):
            # learning conversion rates
            # if purchase
            self.beta_parameters[i][int(pulled_arm[i]), 0] = self.beta_parameters[i][int(pulled_arm[i]), 0] + purchases[
                i]
            # else
            self.beta_parameters[i][int(pulled_arm[i]), 1] = self.beta_parameters[i][int(pulled_arm[i]), 1] + (
                        clicks[i] - purchases[i])

            # computing expected_rewards
            self.counter_per_arm[i][int(pulled_arm[i])] += 1
            self.expected_rewards[i][int(pulled_arm[i])] = (self.expected_rewards[i][int(pulled_arm[i])] * (
                        self.counter_per_arm[i][int(pulled_arm[i])] - 1) + reward[i]) / self.counter_per_arm[i][
                                                               int(pulled_arm[i])]

            # learning the number of bought products
            if purchases[i] != 0:
                self.lambda_poisson[i][int(pulled_arm[i])] = (self.lambda_poisson[i][int(pulled_arm[i])] * (
                            self.counter_per_arm[i][int(pulled_arm[i])] - 1) + daily_units[i] / purchases[i]) / \
                                                             self.counter_per_arm[i][int(pulled_arm[i])] if \
                self.lambda_poisson[i][int(pulled_arm[i])] > 0 else daily_units[i] / purchases[i]
