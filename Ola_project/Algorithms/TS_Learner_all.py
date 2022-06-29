from Algorithms.Learner_Environment_all import *

class TS_all(Learner_all):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.array((np.ones((n_arms, 2)),np.ones((n_arms, 2)),np.ones((n_arms, 2)),np.ones((n_arms, 2)),np.ones((n_arms, 2))))       
        self.expected_rewards = np.zeros([5,n_arms])

    def pull_arm(self, margins_matrix):
        idx = np.zeros(5)
        for i in range(5):
            idx[i] = np.argmax(np.random.beta(self.beta_parameters[i][:, 0], self.beta_parameters[i][:, 1]) * margins_matrix[i,:])
        return idx

    def update(self, pulled_arm, reward, clicks, purchases):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        for i in range(5):
            self.beta_parameters[i][int(pulled_arm[i]), 0] = self.beta_parameters[i][int(pulled_arm[i]), 0] + purchases[i]
            self.beta_parameters[i][int(pulled_arm[i]), 1] = self.beta_parameters[i][int(pulled_arm[i]), 1] + (clicks[i] - purchases[i])
            self.expected_rewards[i][int(pulled_arm[i])] = (self.expected_rewards[i][int(pulled_arm[i])] * (self.t-1) + reward[i]) / self.t