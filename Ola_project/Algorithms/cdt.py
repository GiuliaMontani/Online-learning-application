import math


class CDT:
    def __init__(self, ):
        self.change_estimate = 0


class CUSUM:
    def __init__(self, M, epsilon, threshold, increase):
        """ CUSUM algorithm : see change_detection.pdf algorithm 2
        :param M: CUSUM initialization steps used for computing the mean of the current sample distribution
        :param epsilon: minimum expected mean variation
        :param threshold: threshold for the CUSUM walk
        :param increase: whether to monitor increases or decreases in the mean
        """
        self.change_estimate = 0
        self.M = M
        self.epsilon = epsilon
        self.threshold = threshold
        #self.gaussian = gaussian # gaussian or bernoulli
        self.increase = increase # increase or decrease
        self.reset(0)

    def reset(self, mode):
        if mode == 0:
            self.mean_over_M = 0
            self.num_rewards = 0
        self.g = 0
        self.cumul = 0
        self.min_cumul = 0
        self.change_estimate = self.num_rewards + 1

    def run(self, reward):
        """Run CUSUM algorithm

        :param reward: the new datum that must be analysed by the CDT
        :return: a CDT_Result that contains the alarm and the estimated timestep change
        """

        # Update reward mean
        self.num_rewards += 1
        if self.num_rewards <= self.M:
            self.mean_over_M += reward

        if self.num_rewards == self.M:
            self.mean_over_M /= self.M

        if self.num_rewards <= self.M:
            self.change_estimate = self.num_rewards + 1
            return False
        else:
            s = 0
            '''
            if self.gaussian:
                if self.increase:
                    s = reward - self.mean_over_M - self.epsilon
                else:
                    s = self.mean_over_M - reward - self.epsilon

            else:  # bernoulli
            '''
            if reward > 0.5:  # i.e.reward = 1
                if self.increase:
                    s = math.log(1 + self.epsilon / self.mean_over_M)
                else:
                    s = math.log(1 - self.epsilon / self.mean_over_M)

            else:  # i.e.reward = 0
                if self.increase:
                    s = math.log(1 - self.epsilon / (1 - self.mean_over_M))
                else:
                    s = math.log(1 + self.epsilon / (1 - self.mean_over_M))

        self.g = max(0., self.g + s)
        self.cumul = self.cumul + s
        if self.cumul <= self.min_cumul:
            self.min_cumul = self.cumul
            self.change_estimate = self.num_rewards + 1

        return self.g > self.threshold
