from Algorithms.Learner_Environment import *

class Greedy(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros([5,n_arms])
        self.cr = np.zeros(5)
        #self.greedy_rewards = np.zeros([5,n_arms])

    def pull_arm(self, margins_matrix):
        if self.t < self.n_arms:
            return np.array([self.t] * 5)
        idx = np.zeros(5)
        for i in range(5):
            #idxs = np.argwhere(self.greedy_rewards[i] == self.greedy_rewards[i].max()).reshape(-1)
            idxs = np.argwhere(self.expected_rewards[i] == self.expected_rewards[i].max()).reshape(-1)
            idx[i] = np.random.choice(idxs)
            idx[i] = np.argmax(self.cr[i] * margins_matrix[i,:])
        return idx

    def update(self, pulled_arm, reward, clicks, purchases):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.cr = purchases/clicks
        for i in range(5):
            self.expected_rewards[i][int(pulled_arm[i])] = (self.expected_rewards[i][int(pulled_arm[i])] * (self.t-1) + reward[i]) / self.t
            #self.greedy_rewards[i][int(pulled_arm[i])] = reward[i]