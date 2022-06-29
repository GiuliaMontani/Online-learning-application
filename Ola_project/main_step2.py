# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:55:13 2022

@author: giuli
"""

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

# Step 2: Optimization algorithm
from Algorithms.Greedy_algorithm import *
import matplotlib.pyplot as plt

#matrix with the difference between prices and costs
#mij margins on the price j for the product i
margins = np.zeros((5,4))

for i in range(5):
    margins[i,:] = np.array(products[i].margins_list)
    

#Conversion rate matrix
#in this step all conversion rates are known
#pij conversion rate of price j for the product i
#with higer price (j+1>j) -> lower conversion rate
p1 = np.array([0.38, 0.16, 0.15, 0.1])
p2 = np.array([0.42, 0.41, 0.18, 0.12])
p3 = np.array([0.32, 0.28, 0.17, 0.13])
p4 = np.array([0.36, 0.33, 0.25, 0.18])
p5 = np.array([0.30, 0.29, 0.22, 0.15])
C = np.array([p1,p2,p3,p4,p5])

# rewards per price
#Conversion rate matrix
#in this step all conversion rates are known
#pij conversion rate of price j for the product i
#with higer price (j+1>j) -> lower conversion rate
R = C*margins
R

# computation of the optimal configuration
opt = 0
for i in range(5):
    opt += np.max(R[i,:])
opt

lista = Greedy_algorithm(np.array(C), np.array(margins)) 
max_found = lista[0]
products_to_increase = lista[1]
rewards_per_configuration = lista[2]
max_reward_history = lista[3]
num_it = lista[4]

 #maximization of the cumulative expected margin over all the products
plt.ylabel("Objective function")
plt.plot(range(len(max_reward_history)),max_reward_history, 'r')
plt.plot(range(len(max_reward_history)),np.ones(len(max_reward_history))*opt)
plt.legend(["max_reward_history", "optimal solution"])
#we can see that greedy algorithm not always reach the optimal solution

plt.plot(rewards_per_configuration, 'g')
plt.plot(np.ones(len(rewards_per_configuration))*opt, 'b')
plt.legend(["reward_tested_configurations","optimal configuration"])

# If we test in each day a new configuration we get this regret (over a period of X days) and we never reach the optimal solution
plt.ylabel("Cumulative Regret")
plt.plot(np.cumsum(opt - rewards_per_configuration, axis=0), 'g')
#regret over 100 days
rewards_greedy = np.ones(100-len(rewards_per_configuration))*np.max(rewards_per_configuration)
plt.plot(range(len(rewards_per_configuration),100),np.sum(opt - rewards_per_configuration, axis=0)+np.cumsum(opt - rewards_greedy, axis=0), 'g')