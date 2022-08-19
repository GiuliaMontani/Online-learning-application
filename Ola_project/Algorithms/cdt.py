import math


class CDT:
    def __init__(self, ):
        self.change_estimate = 0


class CUSUM:
    def __init__(self, M, epsilon, threshold):
        """ CUSUM algorithm : cumulative sum algorithm, see change_detection.pdf algorithm 2
        :param M: number of initialization steps used for computing the mean of the current sample distribution
        :param epsilon: minimum expected mean variation
        :param threshold: threshold for the CUSUM walk

        :return 0 if no detection

        """
        self.change_estimate = 0
        self.M = M
        self.epsilon = epsilon
        self.threshold = threshold
        # self.gaussian = gaussian # gaussian or bernoulli
        self.reset(0)

    def reset(self, mode):
        if mode == 0:
            self.mean_over_M = 0
            self.n_rewards = 0
        self.g_increase = 0
        self.g_decrease = 0
        self.cumul_increase = 0
        self.cumul_decrease = 0
        self.min_cumul_increase = 0
        self.min_cumul_decrease = 0
        self.change_estimate_increase = self.n_rewards + 1
        self.change_estimate_decrease = self.n_rewards + 1


    def update(self, reward):
        """Run CUSUM algorithm

        :param reward that has to be analysed by the CDT
        :return: a CDT_Result that contains the alarm and the estimated timestep change
        """

        self.n_rewards += 1
        if self.n_rewards <= self.M:  # if the step time is smaller than n. of initialization steps
            self.mean_over_M += reward

        if self.n_rewards == self.M:
            self.mean_over_M /= self.M

        if self.n_rewards <= self.M:
            self.change_estimate = self.n_rewards + 1
            return False  # too early to detect something

        else:  # if i have enough data to say something
            s_increases = 0
            s_decrease = 0

            # if self.gaussian:
            #    if self.increase:
            #        s = reward - self.mean_over_M - self.epsilon
            #    else:
            #        s = self.mean_over_M - reward - self.epsilon

            # else:  # bernoulli

            if reward > 0.5:  # i.e.reward = 1
                s_increase = math.log(1 + self.epsilon / self.mean_over_M)
                s_decrease = math.log(1 - self.epsilon / self.mean_over_M)

            else:  # i.e.reward = 0
                s_increase = math.log(1 - self.epsilon / (1 - self.mean_over_M))
                s_decrease = math.log(1 + self.epsilon / (1 - self.mean_over_M))

            '''
            for i in range(n_buy):
                s_plus = (1 - self.reference) - self.eps
                s_minus = -(1 - self.reference) - self.eps
                self.g_plus = max(0, self.g_plus + s_plus)
                self.g_minus = max(0, self.g_minus + s_minus)
            for j in range(n_cliks - n_buy):
                s_plus = (0 - self.reference) - self.eps
                s_minus = -(0 - self.reference) - self.eps
                self.g_plus = max(0, self.g_plus + s_plus)
                self.g_minus = max(0, self.g_minus + s_minus)
            '''

        self.g_increase = max(0., self.g_increase + s_increase)
        self.g_decrease = max(0., self.g_decrease + s_decrease)
        self.cumul_increase = self.cumul_increase + s_increase
        self.cumul_decrease = self.cumul_decrease + s_decrease
        if self.cumul_increase <= self.min_cumul_increase:
            self.min_cumul_increase = self.cumul_increase
            self.change_estimate_increase = self.n_rewards + 1

        if self.cumul_decrease <= self.min_cumul_decrease:
            self.min_cumul_decrease = self.cumul_decrease
            self.change_estimate_decrease = self.n_rewards + 1

        return self.g_increase > self.threshold or self.g_decrease > self.threshold
