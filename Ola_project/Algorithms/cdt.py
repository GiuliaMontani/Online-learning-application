import math
import numpy as np

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

    def reset(self, mode=1):
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


    def run(self, reward):
        """Run CUSUM algorithm

        :param reward that has to be analysed by the CDT
        :return: a CDT_Result that contains the alarm and the estimated timestep change
        """
        #print('running CUSUM algorithm')
        #print('reward: ', reward)

        self.n_rewards += 1
        if self.n_rewards <= self.M:  # if the step time is smaller than n. of initialization steps
            self.mean_over_M += reward

        if self.n_rewards == self.M:
            self.mean_over_M /= self.M

        if self.n_rewards <= self.M:
            self.change_estimate = self.n_rewards + 1
            #print('troppo presto')
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
                print('mean_over_M: ',self.mean_over_M)
                s_increase = math.log(1 + self.epsilon / self.mean_over_M)
                s_decrease = math.log(1 - self.epsilon / self.mean_over_M)

            else:  # i.e.reward = 0
                s_increase = math.log(1 - self.epsilon / (1 - self.mean_over_M))
                s_decrease = math.log(1 + self.epsilon / (1 - self.mean_over_M))

            '''
            for i in range(n_buy):
                s_plus = (1 - self.threshold) - self.epsilon
                s_minus = -(1 - self.threshold) - self.eps
                self.g_plus = max(0, self.g_plus + s_plus)
                self.g_minus = max(0, self.g_minus + s_minus)
            for j in range(n_cliks - n_buy):
                s_plus = (0 - self.threshold) - self.eps
                s_minus = -(0 - self.threshold) - self.eps
                self.g_plus = max(0, self.g_plus + s_plus)
                self.g_minus = max(0, self.g_minus + s_minus)
            '''

        self.g_increase = max(0., self.g_increase + s_increase)
        self.g_decrease = max(0., self.g_decrease + s_decrease)

        self.cumul_increase = self.cumul_increase + s_increase
        self.cumul_decrease = self.cumul_decrease + s_decrease
        print('cumul_decrease', self.cumul_decrease)

        if self.cumul_increase <= self.min_cumul_increase:
            self.min_cumul_increase = self.cumul_increase
            self.change_estimate_increase = self.n_rewards + 1

        if self.cumul_decrease <= self.min_cumul_decrease:
            self.min_cumul_decrease = self.cumul_decrease
            self.change_estimate_decrease = self.n_rewards + 1

        return self.g_increase > self.threshold or self.g_decrease > self.threshold


class CUSUM2:
    def __init__(self, M, epsilon, threshold):
        """ CUSUM algorithm : cumulative sum algorithm, see change_detection.pdf algorithm 2
        :param M: number of initialization steps used for computing the mean of the current sample distribution
        :param epsilon: minimum expected mean variation
        :param threshold: threshold for the CUSUM walk

        :return 0 if no detection

        """
        self.t = 0
        self.M = M
        self.epsilon = epsilon
        self.threshold = threshold
        self.mean_over_M = 0
        self.stima = []

        self.g_increase = 0
        self.g_decrease = 0
        self.n_rewards = 0

    def reset(self, mode=1):
        self.g_increase = 0
        self.g_decrease = 0
        self.n_rewards = 0


    def run(self, n_buy, n_cliks):
        """Run CUSUM algorithm

        :param reward that has to be analysed by the CDT
        :return: a CDT_Result that contains the alarm and the estimated timestep change
        """
        print('running CUSUM algorithm')
        self.n_rewards += 1
        if self.n_rewards <= self.M:  # if the step time is smaller than n. of initialization steps
            self.mean_over_M += n_buy/n_cliks
            self.stima.append(n_buy/n_cliks)

        if self.n_rewards == self.M:
            self.mean_over_M /= self.M
            print("the mean over M steps is:", self.mean_over_M)
            print("std:",np.std(self.stima))
            #self.epsilon =np.std(self.stima)*3

        if self.n_rewards <= self.M:
            self.change_estimate = self.n_rewards + 1
            print('troppo presto')
            return False  # too early to detect something

        else:  # if i have enough data to say something
            s_increases = 0
            s_decrease = 0

            print('mean_over_M: ', self.mean_over_M)
            for i in range(n_buy):
                s_increase = (1 - self.mean_over_M) - self.epsilon #(1 - average value we should obtain) - accepted variation
                s_decrease = -(1 - self.mean_over_M) - (self.epsilon*5)
                self.g_increase = max(0, self.g_increase + s_increase)
                self.g_decrease = max(0, self.g_decrease + s_decrease)

            for j in range(n_cliks - n_buy):
                s_increase = (0 - self.mean_over_M) - self.epsilon
                s_decrease = -(0 - self.mean_over_M) - (self.epsilon*5)
                self.g_increase = max(0, self.g_increase + s_increase)
                self.g_decrease = max(0, self.g_decrease + s_decrease)

            print('g_increase',self.g_increase)
            print('g_decrease',self.g_decrease)

        return self.g_increase > self.threshold or self.g_decrease > self.threshold
