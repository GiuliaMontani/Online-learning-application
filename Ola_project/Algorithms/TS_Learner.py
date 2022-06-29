from Algorithms.Learner_Environment import *

class TS(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self, margins_list):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * margins_list)
        return idx

    def update(self, pulled_arm, reward, clicks, purchases):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + purchases
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + (clicks - purchases)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t-1) + reward) / self.t