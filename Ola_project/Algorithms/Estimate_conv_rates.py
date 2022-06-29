from Algorithms.Learner_Environment import *
import random

class Conv_rates(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.expected_rewards = np.zeros(n_arms)
        self.i = -1

    def pull_arm(self):
        self.i += 1
        if self.i >3:
            self.i = 0
        return np.random.choice(4)#self.i

    def update(self, pulled_arm, reward, clicks, purchases):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + purchases
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + (clicks - purchases)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t-1) + reward) / self.t