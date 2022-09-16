from Algorithms.Greedy_algorithm import *
import matplotlib.pyplot as plt
from Environment.E_commerce import *

# %%
P1 = Product(0, [9, 12, 13, 14.5], [1., 4, 5., 6.5])
P2 = Product(1, [20, 22.5, 23, 24.5], [4., 6.5, 7., 8.5])
P3 = Product(2, [30, 31.5, 34, 34.5], [6., 7.5, 10., 10.5])
P4 = Product(3, [40, 42.5, 43, 46.5], [8., 10.5, 11., 14.5])
P5 = Product(4, [50, 51.5, 53, 54.5], [10., 11.5, 13., 14.5])

products = [P1, P2, P3, P4, P5]

E = E_commerce()
E.graph = np.array(
    [[0., 0.5, 0., 1., 0.], [1., 0., 0., 0.5, 0.], [0., 1., 0., 0.5, 0.], [0., 0.5, 1., 0., 0.], [1., 0., 0., 0.5, 0.]])
E.set_products(products)
E.set_lambda(0.5)
# %%
# matrix with the difference between prices and costs
# m_ij margins on the price j for the product i
margins_matrix = np.zeros((5, 4))
for i in range(5):
    for j in range(4):
        margins_matrix[i, j] = products[i].margins_list[j]

margins = margins_matrix
# Conversion rate matrix
# in this step all conversion rates are known
# p_ij conversion rate of price j for the product i
# with higher price (j+1>j) -> lower conversion rate
p1 = np.array([0.38, 0.16, 0.15, 0.1])
p2 = np.array([0.42, 0.41, 0.18, 0.12])
p3 = np.array([0.32, 0.28, 0.17, 0.13])
p4 = np.array([0.36, 0.33, 0.25, 0.18])
p5 = np.array([0.30, 0.29, 0.22, 0.15])
C = np.array([p1, p2, p3, p4, p5])
# They are supposed to be known, but we can take their estimations from step 3
C = np.array(
    [[0.69334299, 0.15864193, 0.06566728, 0.01168772],
     [0.93377632, 0.59672407, 0.50127129, 0.22567331],
     [0.99395335, 0.95952215, 0.691365, 0.59561344],
     [0.69047299, 0.22901752, 0.15824826, 0.00297488],
     [0.97734534, 0.89545417, 0.68816697, 0.40171434]])

# rewards per price
# Conversion rate matrix
# in this step all conversion rates are known
# p_ij conversion rate of price j for the product i
# with higer price (j+1>j) -> lower conversion rate
R = C * margins

print(R)
# %%
C[2,2] = 0.9
R = C * margins
print(R)
print(0.9*8>7.19641612*0.95952215)



# %%
# computation of the optimal configuration
opt = 0
for i in range(5):
    opt += np.max(R[i, :])

best_arm_per_product = np.zeros(5)
for i in range(5):
    best_arm_per_product[i] = np.argmax(np.array(E.products[i].margins_list) * C[i, :])

print("Optimal expected reward:", opt)
print("Best configuration: ", best_arm_per_product) #best arm for product i

lista = Greedy_algorithm(np.array(C), np.array(margins))
max_found = lista[0]
products_to_increase = lista[1]
rewards_per_configuration = lista[2]
max_reward_history = lista[3]
num_it = lista[4]
print(C*margins)
print(rewards_per_configuration)
print(products_to_increase)

