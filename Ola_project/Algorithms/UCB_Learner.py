from Algorithms.Learner_Environment import *

class UCB(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)

    def pull_arm(self):
        first_pulls = [0,1,2,3,0,1,2,3]
        if self.t < self.n_arms:
            return self.t
        upper_conf = self.empirical_means + self.confidence
        return np.random.choice(np.argwhere(upper_conf == upper_conf.max()).reshape(-1))

    def update(self, pulled_arm, reward):
        self.t += 1
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t-1)  + reward) / self.t
        n_samples = np.size(self.rewards_per_arm[pulled_arm])
        self.confidence[pulled_arm] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
        self.update_observations(pulled_arm, reward)