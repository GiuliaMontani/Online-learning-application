from Algorithms.UCB_Learner import *

# for context generation
class CG_UCB(Learner):
    def __init__(self, n_arms, c=2):
        """UCB Learner algorithm.

        :param n_arms: number of arms
        :param c: confidence value (in class it was 2)
        """
        super().__init__(n_arms)
        self.expected_rewards_per_type = [[np.zeros([5, n_arms]) for _ in range(2)] for _ in range(2)]
        self.confidence_per_type = [[np.array(([[np.inf] * n_arms] * 5)) for _ in range(2) ] for _ in range(2)]
        self.make_comparable_per_type = [[np.zeros(5) for _ in range(2)] for _ in range(2) ]
        self.counter_per_arm_per_type = [[np.array([np.zeros(4)] * 5) for _ in range(2)] for _ in range(2)]
        self.explore = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        self.c = c
        self.split_1 = 0
        self.split_2 = 0

    def pull_arm(self):

        upper_conf_per_type = [[np.zeros((5, 4)) for _ in range(2)] for _ in range(2)]

        # split_1 == 0 and split_2 == 0 --> don't split (one class)
        # split_1 == 1 and split_2 == 0 --> split into User0 and User1+User2 (2 classes)
        # split_1 == 0 and split_2 == 1 --> split into User0+User1 and User2 (2 classes)
        # split_1 == 1 and split_2 == 1 --> split into User0, User1 and User2 (3 classes)

        if self.t < 2 * self.n_arms:  # --> why 2??
            return np.array([self.explore[self.t]] * 5)  # for each product *5, play the t-th price

        # at the end of the first exploration phase we compute the maximum expected reward for each product,
        # we store and use it to compare the upper confidence
        if self.t == 2 * self.n_arms:  # 2*4=8
            for i in range(5):
                self.make_comparable_per_type[self.self.split_1][self.split_2][i] = np.std(self.expected_rewards_per_type[self.self.split_1][self.split_2][i])  # sd. var. of first collected rewards

        idx = np.zeros(5)
        upper_conf = np.zeros((5, 4))

        # chose the arm with the highest expected_rewards for each product,
        # taking into account the confidence interval (which gets smaller more time the corresponding arm is pulled)
        for i in range(5):
            # confidence is multiplied by np.max(self.expected_rewards[i]) to make the upper confidence bound
            # comparable (some have too much higher expected rewards) ???
            upper_conf_per_type[self.split_1][self.split_2][i] = self.expected_rewards_per_type[self.split_1][self.split_2][i] + self.confidence_per_type[self.split_1][self.split_2][i] * self.make_comparable_per_type[self.split_1][self.split_2][i]
            # increasing the term make_comparable we get a longer exploration phase, but we have more accuracy in
            # choosing the best candidates

            # idx[i] = np.argmax(upper_conf[i]) #but in case of two maximum
            idx[self.split_1][self.split_2][i] = np.random.choice(np.argwhere(upper_conf_per_type[self.split_1][self.split_2][i] == upper_conf_per_type[self.split_1][self.split_2][i].max()).reshape(-1))  # 2x slower

        return idx

    def update(self, pulled_arm, reward):
        self.t += 1

        for self.split_1 in range(2):
            for self.split_2 in range(2):
                for i in range(5):
                    self.counter_per_arm_per_type[self.split_1][self.split_2][i][int(pulled_arm[self.split_1][self.split_2][i])] += 1
                    self.expected_rewards_per_type[self.split_1][self.split_2][i][int(pulled_arm[self.split_1][self.split_2][i])] = (self.expected_rewards_per_type[self.split_1][self.split_2][i][int(pulled_arm[self.split_1][self.split_2][i])] * (
                            self.counter_per_arm_per_type[self.split_1][self.split_2][i][int(pulled_arm[self.split_1][self.split_2][i])] - 1) + reward[i]) / self.counter_per_arm_per_type[self.split_1][self.split_2][i][
                                                                       int(pulled_arm[self.split_1][self.split_2][i])]
                    n_samples = np.size(self.rewards_per_arm[i][int(pulled_arm[i])])
                    self.confidence_per_type[i][int(pulled_arm[i])] = (self.c * np.log(
                        self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf

                self.update_observations(pulled_arm[self.split_1][self.split_2], reward)
