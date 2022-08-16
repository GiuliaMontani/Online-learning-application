from Algorithms.Learner_Environment import *

from Algorithms.Learner_Environment import *


class UCB(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros([5, n_arms])
        self.confidence = np.array(([[np.inf] * n_arms] * 5))
        self.make_comparable = np.zeros(5)
        self.explore = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        self.c = 2

    def pull_arm(self):
        # first exploration phase
        if self.t < 2 * self.n_arms:  # --> why 2??
            return np.array([self.explore[self.t]] * 5)  # for each product *5, play the t-th price

        # at the end of the first exploration phase we compute the maximum expected reward for each product,
        # we store and use it to compare the upper confidence
        if self.t == 2 * self.n_arms:  # 2*4=8
            for i in range(5):
                self.make_comparable[i] = np.std(self.expected_rewards[i])  # sd. var. of first collected rewards

        idx = np.zeros(5)
        upper_conf = np.zeros((5, 4))

        # chose the arm with the highest expected_rewards for each product,
        # taking into account the confidence interval (which gets smaller more time the corresponding arm is pulled)
        for i in range(5):
            # confidence is multiplied by np.max(self.expected_rewards[i]) to make the upper confidence bound
            # comparable (some have too much higher expected rewards) ???
            upper_conf[i] = self.expected_rewards[i] + self.confidence[i] * self.make_comparable[i]
            # increasing the term make_comparable we get a longer exploration phase, but we have more accuracy in
            # choosing the best candidates

            # idx[i] = np.argmax(upper_conf[i]) but in case of two maximum
            idx[i] = np.random.choice(np.argwhere(upper_conf[i] == upper_conf[i].max()).reshape(-1))

        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        for i in range(5):
            self.counter_per_arm[i][int(pulled_arm[i])] += 1
            self.expected_rewards[i][int(pulled_arm[i])] = (self.expected_rewards[i][int(pulled_arm[i])] * (
                    self.counter_per_arm[i][int(pulled_arm[i])] - 1) + reward[i]) / self.counter_per_arm[i][
                                                               int(pulled_arm[i])]
            n_samples = np.size(self.rewards_per_arm[i][int(pulled_arm[i])])
            self.confidence[i][int(pulled_arm[i])] = (self.c * np.log(
                self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf

        self.update_observations(pulled_arm, reward)
