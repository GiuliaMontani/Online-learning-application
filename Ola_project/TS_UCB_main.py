# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:56:31 2022

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
# xij = probabilities starting from product i to see product j
E.graph

# Step 3: Optimization with uncertain conversion rates
# TS
from Algorithms.TS_Learner import *

#Thompson Sampling
n_arms = 4

num_users = 1000
binary_vector = np.array([127,0]) 
n_days = 100
product_index= 0
fixed_alpha = 1


env = Environment(n_arms, E, product_index, num_users, binary_vector, fixed_alpha)
ts_learner = TS(n_arms=n_arms)
for d in range(n_days):
    E.simulate_day(num_users, binary_vector, 1)
    
    pulled_arm = ts_learner.pull_arm(env.margins_list)
    reward = env.round(pulled_arm)
    ts_learner.update(pulled_arm, reward, env.clicks_current_day, env.purchases_current_day)

print("end")
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum((opt - ts_learner.collected_rewards)), 'r')
plt.legend(["TS"])
plt.show()

plt.plot(ts_learner.collected_rewards)
np.mean(ts_learner.collected_rewards)

print("mean rewards per arm")
print(ts_learner.expected_rewards)
print("conversion rates of the samples")
print(ts_learner.beta_parameters[:,0]/(ts_learner.beta_parameters[:,0]+ts_learner.beta_parameters[:,1]))
pulls_per_arm = [len(ts_learner.rewards_per_arm[0]),len(ts_learner.rewards_per_arm[1]),len(ts_learner.rewards_per_arm[2]),len(ts_learner.rewards_per_arm[3])]
print("pulls per arm")
print(pulls_per_arm)

# UCB
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

from Algorithms.UCB_Learner import *

#UCB

n_arms = 4

ucb_rewards_per_experiment = []

num_users = 1000
binary_vector = np.array([127,0]) 
n_days = 100
product_index= 0
fixed_alpha = 1


env = Environment(n_arms, E, product_index, num_users, binary_vector, fixed_alpha)
ucb_learner = UCB(n_arms=n_arms)
for d in range(n_days):
    E.simulate_day(num_users, binary_vector, 1)
    
    pulled_arm = ucb_learner.pull_arm()
    reward = env.round(pulled_arm)
    ucb_learner.update(pulled_arm, reward)


print("end")
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum((opt - ucb_learner.collected_rewards)), 'r')
plt.legend(["UCB"])
plt.show()

plt.plot(ucb_learner.collected_rewards)
np.mean(ucb_learner.collected_rewards)

print("mean rewards per arm")
print(ucb_learner.empirical_means)
print("confidence")
print(ucb_learner.confidence)
pulls_per_arm = [len(ucb_learner.rewards_per_arm[0]),len(ucb_learner.rewards_per_arm[1]),len(ucb_learner.rewards_per_arm[2]),len(ucb_learner.rewards_per_arm[3])]
print("pulls per arm")
print(pulls_per_arm)
#Comparison between UCB and TS
opt = np.max(np.array(E.products[0].margins_list) * conversion_rates)
print("end")
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum((opt - ucb_learner.collected_rewards)), 'b')
plt.plot(np.cumsum((opt - ts_learner.collected_rewards)), 'r')
plt.legend(["UCB","TS"])
plt.show()

np.sum(ucb_learner.collected_rewards), np.sum(ts_learner.collected_rewards)

print("UCB reward: ",np.sum(ucb_learner.collected_rewards))
print("TS reward: ", np.sum(ts_learner.collected_rewards))