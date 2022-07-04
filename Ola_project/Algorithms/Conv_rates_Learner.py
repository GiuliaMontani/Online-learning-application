from Algorithms.Learner_Environment import *
import random

class Conv_rates(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.array((np.ones((n_arms, 2)),np.ones((n_arms, 2)),np.ones((n_arms, 2)),np.ones((n_arms, 2)),np.ones((n_arms, 2))))       
        self.expected_rewards = np.zeros([5,n_arms])

    def pull_arm(self, margins_matrix):
        if self.t < self.n_arms:
            return np.array([self.t] * 5)        
        return np.random.choice(4,size=5)

    def update(self, pulled_arm, reward, clicks, purchases):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        for i in range(5):
            self.beta_parameters[i][int(pulled_arm[i]), 0] = self.beta_parameters[i][int(pulled_arm[i]), 0] + purchases[i]
            self.beta_parameters[i][int(pulled_arm[i]), 1] = self.beta_parameters[i][int(pulled_arm[i]), 1] + (clicks[i] - purchases[i])
            self.expected_rewards[i][int(pulled_arm[i])] = (self.expected_rewards[i][int(pulled_arm[i])] * (self.t-1) + reward[i]) / self.t