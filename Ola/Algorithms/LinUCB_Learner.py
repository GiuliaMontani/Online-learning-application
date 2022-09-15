from Algorithms.Learner_Environment import *

class LinUCB_Learner(Learner):

    def __init__(self, n_arms, arms_features, c=2):
        """Linear UCB Learner algorithm.

            :param n_arms: number of arms
            :param arms_features: features for each arm
            """
        super().__init__(n_arms)
        self.arms = arms_features
        self.dim = arms_features.shape[1]
        self.expected_rewards = np.zeros([5, n_arms])
        self.confidence = np.array(([[np.inf] * n_arms] * 5))
        self.make_comparable = np.zeros(5)
        self.explore = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        self.c = c
        self.M = np.identity(self.dim)
        self.b = np.atleast_2d(np.zeros(self.dim)).T   # zero vector of dimension dim
        self.theta = np.dot(np.linalg.inv(self.M), self.b) # initialize theta

    def compute_ucbs(self):
        self.theta = np.dot(np.linalg.inv(self.M), self.b)
        ucbs = [[]]
        for arm in self.arms:
            arm = np.atleast_2d(arm).T
            ucb = np.dot(self.theta, arm) + self.c + np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.M), arm)))
            ucbs.append(ucb)
         return ucbs

    def pull_arm(self):
        ucbs = self.compute_ucbs()
        return np.argmax(ucbs)

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

            # idx[i] = np.argmax(upper_conf[i]) #but in case of two maximum
            idx[i] = np.random.choice(np.argwhere(upper_conf[i] == upper_conf[i].max()).reshape(-1))  # 2x slower

        return idx


    def update_observations(self, pulled_arm, reward):



    def update(self, arm_idx, reward):


