# fixed graph weights (same click influence probabilities between the products for each user)
# homogeneous users (one class with small variability)
# fixed alphas (same number of users each day distributed equally between the initial webpages)
# uncertain conversion rates
# %%
import matplotlib.pyplot as plt
import numpy as np
from Environment.E_commerce import *
from Algorithms.Conv_rates_Learner import *
from Algorithms.Greedy_Learner import *
from Algorithms.TS_Learner import *
from Algorithms.UCB_Learner import *

# %%
P1 = Product(0, [9, 12, 13, 14.5], [1., 4, 5., 6.5])
P2 = Product(1, [20, 22.5, 23, 24.5], [4., 6.5, 7., 8.5])
P3 = Product(2, [30, 31.5, 34, 34.5], [6., 7.5, 10., 10.5])
P4 = Product(3, [40, 42.5, 43, 46.5], [8., 10.5, 11., 14.5])
P5 = Product(4, [50, 51.5, 53, 54.5], [10., 11.5, 13., 14.5])

products = [P1, P2, P3, P4, P5]
margins_matrix = np.zeros((5, 4))
for i in range(5):
    for j in range(4):
        margins_matrix[i, j] = products[i].margins_list[j]
print("Margin matrix: ")
print(margins_matrix)

E = E_commerce()
E.graph = np.array(
    [[0., 0.5, 0., 1., 0.], [1., 0., 0., 0.5, 0.], [0., 1., 0., 0.5, 0.], [0., 0.5, 1., 0., 0.], [1., 0., 0., 0.5, 0.]])
E.set_products(products)
E.set_lambda(0.5)
# %% md
# Estimation of conversion rates and expected rewards of the arms for each product to compute the clairvoyant solution

# Random algorithm which for each round pulls a random choice
# to estimate asymptotically the conv_rates and the mean of the number of units sold per product,
# useful for computing clairvoyant solution and regrets of the bandit algorithms

n_arms = 4
num_users = 10  # mean number of users for each day 1000
binary_vector = np.array([127, 0])
n_days = 5  # 1000
fixed_alpha = 1
fixed_weights = 1
fixed_units = 1
num_experiments = 1  # 10
opt_vector = np.zeros(num_experiments)
conv_rates_per_experiment = []
mean_units_sold_per_product_per_experiment = []
cr_learner_expected_rewards_per_experiment = []
gr_learner = Greedy(n_arms=n_arms)

for e in range(num_experiments):
    env = Environment(n_arms, E, margins_matrix, num_users, binary_vector, fixed_alpha, fixed_weights, fixed_units)
    cr_learner = Conv_rates(n_arms=n_arms)

    for d in range(n_days):
        pulled_arm = gr_learner.pull_arm(env.margins_matrix)
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward, env.clicks_current_day, env.purchases_current_day)

    conversion_rates = np.zeros((5, 4))
    np.set_printoptions(suppress=True)
    for i in range(5):
        conversion_rates[i] = cr_learner.beta_parameters[i][:, 0] / (
                    cr_learner.beta_parameters[i][:, 0] + cr_learner.beta_parameters[i][:, 1])
        opt_vector[e] += np.max(
            np.array(E.products[i].margins_list) * conversion_rates[i] * cr_learner.lambda_poisson[i])

    conv_rates_per_experiment.append(conversion_rates)
    mean_units_sold_per_product_per_experiment.append(cr_learner.lambda_poisson)
    cr_learner_expected_rewards_per_experiment.append(cr_learner.expected_rewards)

# optimal expected clarvoyant solution is given choosing each round the best combination
opt = np.mean(opt_vector)  # + np.std(opt_vector)
best_arm_per_product = np.zeros(5)
for i in range(5):
    best_arm_per_product[i] = np.argmax(
        np.array(E.products[i].margins_list) * np.mean(conv_rates_per_experiment, axis=0)[i]
        * np.mean(mean_units_sold_per_product_per_experiment, axis=0)[i])  # expected_units_sold_per_product[i])#

print("_______________________________________________")
print("Conversion rates")
print(np.mean(conv_rates_per_experiment, axis=0))
print("_______________________________________________")
print("Expected rewards per arm")
print(np.mean(cr_learner_expected_rewards_per_experiment, axis=0))
print("_______________________________________________")
print("Expected units sold per arm")
print(np.mean(mean_units_sold_per_product_per_experiment, axis=0))

print("_______________________________________________")
print("Best configuration", best_arm_per_product)
print("Optimal cumulative expected reward per round")
print(opt)  # optimal configuration: the best combination of arms

# %% md
i=0
idx = np.zeros(5)
upper_conf = np.array(
    [[0., 0.5, 0., 1.], [1., 0., 0., 0.5], [0., 1., 0., 0.5], [0., 0.5, 1., 0.], [1., 0., 0., 0.5]])
idx[i] = np.random.choice(np.argwhere(upper_conf[i] == upper_conf[i].max()).reshape(-1))