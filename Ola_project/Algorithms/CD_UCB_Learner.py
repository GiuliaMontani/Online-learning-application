from Algorithms.UCB_Learner import *


class CD_UCB(UCB):
    def __init__(self, n_arms, c=2, epsilon=0.2):
        """Change-Detection based UCB (CD-UCB). Look at change_detection.pdf
        I am not sure to have really understood paper's equation 5
        :param n_arms:
        :param c:
        :param epsilon:
        """
        super().__init__(n_arms, c)
        self.epsilon = epsilon

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
            # probability of selecting a random price
            p = np.random.random()
            if p < self.epsilon:
                idx[i] = np.random.choice(self.n_arms)
            else:
            # confidence is multiplied by np.max(self.expected_rewards[i]) to make the upper confidence bound
            # comparable (some have too much higher expected rewards) ???
                upper_conf[i] = self.expected_rewards[i] + self.confidence[i] * self.make_comparable[i]
                # increasing the term make_comparable we get a longer exploration phase, but we have more accuracy in
                # choosing the best candidates

                # idx[i] = np.argmax(upper_conf[i]) #but in case of two maximum
                idx[i] = np.random.choice(np.argwhere(upper_conf[i] == upper_conf[i].max()).reshape(-1))  # 2x slower

        return idx