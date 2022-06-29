from Algorithms.Learner_Environment_all import *

class UCB_all(Learner_all):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros([5,n_arms])
        self.confidence = np.array(([np.inf] * n_arms,[np.inf] * n_arms,[np.inf] * n_arms,[np.inf] * n_arms,[np.inf] * n_arms))

    def pull_arm(self):
        if self.t < self.n_arms:
            return np.array([self.t] * 5)
        upper_conf = self.expected_rewards + self.confidence
        idx = np.zeros(5)
        for i in range(5):
            idx[i] = np.random.choice(np.argwhere(upper_conf[i] == upper_conf[i].max()).reshape(-1))
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        for i in range(5):
            self.expected_rewards[i][int(pulled_arm[i])] = (self.expected_rewards[i][int(pulled_arm[i])]*(self.t-1)  + reward[i]) / self.t
            n_samples = np.size(self.rewards_per_arm[i][int(pulled_arm[i])])
            self.confidence[i][int(pulled_arm[i])] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        self.update_observations(pulled_arm, reward)