# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 19:08:23 2022

@author: giuli
"""
import matplotlib.pyplot as plt
from Environment.E_commerce import *
# Hp: the costs are 80% of the lower prices

P1 = Product(0,[10,13,16,19],[ 2.,  5.,  8., 11.])
P2 = Product(1,[20,23,26,29],[ 4.,  7., 10., 13.])
P3 = Product(2,[30,33,36,39],[ 6.,  9., 12., 15.])
P4 = Product(3,[40,43,46,49],[ 8., 11., 14., 17.])
P5 = Product(4,[50,53,56,59],[10., 13., 16., 19.])

products = [P1,P2,P3,P4,P5]
# E_commerce inizialization
E = E_commerce()
E.set_products(products)
E.set_lambda(0.5)

from Algorithms.TS_Learner_all import *

#Thompson Sampling
# n_arms = 4

# num_users = 1000
# binary_vector = np.array([127,0]) 
# n_days = 10
# fixed_alpha = 1
# margins_matrix = np.array([[2.,  5.,  8., 11.], [4.,  7., 10., 13.],[6.,  9., 12., 15.],[ 8., 11., 14., 17],[10., 13., 16., 19.]])

# env = Environment_all(n_arms, E, margins_matrix, num_users, binary_vector, fixed_alpha)
# ts_learner = TS_all(n_arms=n_arms)
# for d in range(n_days):
#     E.simulate_day(num_users, binary_vector, 1)
    
#     pulled_arm = ts_learner.pull_arm(env.margins_matrix)
#     reward = env.round(pulled_arm)
#     ts_learner.update(pulled_arm, reward, env.clicks_current_day, env.purchases_current_day)


# opt_ts = np.max((ts_learner.collected_rewards))
# print("end")
# plt.figure(0)
# plt.xlabel("t")
# plt.ylabel("Regret")
# plt.plot(np.cumsum((opt_ts - ts_learner.collected_rewards)), 'r')
# plt.legend(["TS"])
# plt.show()

# ts_learner.collected_rewards

# print("conversion rates of the samples")
# for i in range(5):
#     print(ts_learner.beta_parameters[i][:,0]/(ts_learner.beta_parameters[i][:,0]+ts_learner.beta_parameters[i][:,1]))
    
    
# ts_learner.expected_rewards

# print("pulls per arm")
# for i in range(5):
#     pulls_per_arm = [len(ts_learner.rewards_per_arm[i][0])-1,len(ts_learner.rewards_per_arm[i][1])-1,len(ts_learner.rewards_per_arm[i][2])-1,len(ts_learner.rewards_per_arm[i][3])-1]

#     print(pulls_per_arm)
    
    
# UCB
from Algorithms.UCB_Learner_all import *
n_arms = 4

num_users = 1000
binary_vector = np.array([127,0]) 
n_days = 10
fixed_alpha = 1
margins_matrix = np.array([[2.,  5.,  8., 11.], [4.,  7., 10., 13.],[6.,  9., 12., 15.],[ 8., 11., 14., 17],[10., 13., 16., 19.]])

env = Environment_all(n_arms, E, margins_matrix, num_users, binary_vector, fixed_alpha)
ucb_learner = UCB_all(n_arms=n_arms)
for d in range(n_days):
    E.simulate_day(num_users, binary_vector, 1)
    
    pulled_arm = ucb_learner.pull_arm()
    reward = env.round(pulled_arm)
    ucb_learner.update(pulled_arm, reward)

opt = np.max((ucb_learner.collected_rewards))
print("end")
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum((opt - ucb_learner.collected_rewards)), 'r')
plt.legend(["UCB"])
plt.show()