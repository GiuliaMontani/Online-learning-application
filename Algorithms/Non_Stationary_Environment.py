from Algorithms.Learner_Environment import Environment
import numpy as np
class NonStationaryEnvironment(Environment):
    def _init__(self, n_arms, probabilities, horizon):
        super().__init__(n_arms, probabilities)
        self.t = 0  # initial time
        n_phases=len(self.probabilities)
        self.phases_size = horizon/n_phases

    def round(self, pulled_arm):
        current_phase = int(self.t / self.phase_size)
        p=self.probabilities[current_phase][pulled_arm]
        reward = np.random.binomial(1,p)
        self.t += 1
        return reward
