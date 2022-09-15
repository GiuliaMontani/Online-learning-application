from Algorithms.Learner_Environment import *


class Greedy(Learner):
    def __init__(self, n_arms, epsilon = 0.2):
        """ε-Greedy algorithm: If the random value 'p' is less than Epsilon then a random action will be chosen,
        otherwise the socket with the highest current estimated reward will be selected.

        :param n_arms: number of prices to test
        :param epsilon: average probability of random action (if eps=1, a.s. random choice)
        """
        super().__init__(n_arms)
        self.expected_rewards = np.zeros([5, n_arms])
        self.epsilon = epsilon
        # self.greedy_rewards = np.zeros([5,n_arms])

    def pull_arm(self, margins_matrix):
        #  margins_matrix) not needed
        if self.t < self.n_arms:
            return np.array([self.t] * 5)

        idx = np.zeros(5)
        for i in range(5):
            # probability of selecting a random price
            p = np.random.random()
            if p < self.epsilon:
                idx[i] = np.random.choice(self.n_arms)
            else:
                # idx[i] = np.argmax(self.expected_rewards[i])
                idxs = np.argwhere(self.expected_rewards[i] == self.expected_rewards[i].max()).reshape(-1)
                idx[i] = np.random.choice(idxs)

        return idx

    def update(self, pulled_arm, reward, clicks, purchases):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        for i in range(5):
            self.counter_per_arm[i][int(pulled_arm[i])] += 1
            self.expected_rewards[i][int(pulled_arm[i])] = (self.expected_rewards[i][int(pulled_arm[i])] * (
                        self.counter_per_arm[i][int(pulled_arm[i])] - 1) + reward[i]) / self.counter_per_arm[i][
                                                               int(pulled_arm[i])]
            # self.greedy_rewards[i][int(pulled_arm[i])] = reward[i]
