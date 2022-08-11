import numpy as np
from Environment.Daily_Customers import *
from Environment.Product import *


class E_commerce:
    # constructor
    def __init__(self):
        # list of products
        self.products = []
        # list of lists of users
        self.daily_users = []
        # dataset with the history of products visited by each user in each day
        self.time_history = []
        # probability that the user checks the second product
        self.lambda_ = 0.5
        # weight of the graph with the probabilities to see a secondary product given the primary
        self.graph = np.array(
            [[0., self.lambda_, 0., 1., 0.],
             [1., 0., 0., self.lambda_, 0.],
             [0., 1., 0., self.lambda_, 0.],
             [0., self.lambda_, 1., 0., 0.],
             [1., 0., 0., self.lambda_, 0.]])
        # list of the total daily rewards for each day
        self.daily_rewards = []
        # clicks per product for the current day
        self.daily_clicks = []
        # purchases per product for the current day
        self.daily_purchases = []
        self.daily_rewards_per_product = []
        # daily purchased units per product
        self.daily_purchased_units = []

    def set_lambda(self, new_lambda):
        self.lambda_ = new_lambda

    def add_product(self, product):
        self.products.append(product)

    def set_products(self, product_list):
        self.products = product_list

    def clear_history(self):
        self.time_history.clear()
        self.daily_rewards.clear()
        self.daily_users.clear()

        # USED AT THE BEGINNING BUT NOW IS USELESS

    # graph with the probabilities to see the products
    # 1 for the first secondary slot, lambda for the second one
    # def generate_graph(self, distribution):
    #     graph = np.zeros((5,5))
    #     for i in range(5):
    #       # secondary slots indexes (0,1,2,3,4,5)-{i=primary}
    #       j = np.random.choice([x for x in range(5) if x != i ],2, replace = False) 
    #       # probability to see the first slot = 1
    #       graph[i,j[0]] = distribution[i]
    #       # 1 * lambda 
    #       graph[i,j[1]] = distribution[i] * self.lambda_
    #     return graph

    # simulate a day of visits in the website
    def simulate_day(self, number_users, binary_vector, fixed_alpha, fixed_weights, fixed_units, binary_features=0, alpha=np.ones(5)):
        """This function simulate a day of visits in the website

        :param number_users: average number of potential users in a day
        :type number_users: int
        :param binary_vector: type of user (0,1 or 2)
        :type binary_vector: int
        :param fixed_alpha: 1 if alpha is fixed (uniformly distributed over the products)
        :type fixed_alpha: bool
        :param fixed_weights: 1 if alpha is fixed
        :type fixed_weights: bool
        :param fixed_units: if 1 always the same number of units is bought for each product
        :type fixed_units: bool
        :param binary_features: 1 if we do not distinguish between users (STEP 1)
        :type binary_features: bool
        :param alpha: vector with dirichlet parameters
        :return: a Daily_Customers class

        """

        num_users = int(np.random.normal(number_users, scale=0.2 * number_users, size=1))  # drawn from a gaussian
        D = Daily_Customers()
        D.UsersGenerator(num_users, binary_vector, fixed_alpha, alpha, fixed_weights, binary_features)
        self.daily_users.append(D.Users)

        rewards_of_the_day = 0
        # store the visits of the day
        Day = []
        self.daily_rewards_per_product = np.zeros(5)
        self.daily_purchased_units = np.zeros(5)
        self.daily_purchases = np.zeros(5)
        self.daily_clicks = np.zeros(5)

        # for each user visit (each day we can change the prices: we have to implement it)
        for i in range(np.size(D.Users)):
            visit = self.visit(D.Users[i], fixed_units)
            Day.append(visit)
            # compute and update all the variables that we will use in the algorithms
            for k in range(np.size(D.Users[i].cart)):
                rewards_of_the_day += self.products[
                    D.Users[i].cart[k]].margin  # * self.products[D.Users[i].cart[k]].margin #if we have units!=1
                self.daily_purchases[D.Users[i].cart[k]] += 1
                self.daily_purchased_units[D.Users[i].cart[k]] += D.Users[i].quantities[k]
                self.daily_rewards_per_product[D.Users[i].cart[k]] += D.Users[i].quantities[k] * self.products[
                    D.Users[i].cart[k]].margin
            for z in range(np.size(D.Users[i].products_clicked)):
                self.daily_clicks[D.Users[i].products_clicked[z]] += 1

        self.time_history.append(Day)
        self.daily_rewards.append(rewards_of_the_day)
        D.Users[:] = []
        return Day

    def visit(self, user, fixed_units):
        """ Simulate when a user visits the website.

        :param user: user which visits
        :type user: User
        :param fixed_units: if 1 always the same number of units is bought for each product
        :type fixed_units: bool
        :return:
        """

        # Influence probability matrix of the products, for each user equal to the see probability * click probability
        prob_matrix = user.P * self.graph
        n_nodes = prob_matrix.shape[0]

        # if user's reservation price is lower than the price of the primary product -> end the visit
        if user.reservation_price[user.primary] < self.products[user.primary].price:  # not bought
            user.products_clicked = [user.primary]
            history_purchase = []
            user.cart = []
            user.quantities = []
            active_nodes = np.zeros(n_nodes)
            active_nodes[user.primary] = 1  # [0 1 0 0 0]
            return np.array([active_nodes])

        # Influence probability matrix of the products, for each user equal to the see probability*click probability
        # obtain two products with non-null probabilities that are the secondary product for each row
        # i.e. every product has two secondary product
        prob_matrix = user.P * self.graph
        n_nodes = prob_matrix.shape[0]
        active_nodes = np.zeros(n_nodes)
        active_nodes[user.primary] = 1
        newly_active_nodes = active_nodes
        round = 0

        # store index of products clicked
        history_click = [user.primary]
        # store products bought
        history_purchase = [user.primary]
        # store products shown to the user but maybe not clicked or bought
        history_nodes = np.array([active_nodes])

        # store the prices of the 5 products in an array
        prod_prices = np.zeros(len(self.products))
        for i in range(len(self.products)):
            prod_prices[i] = self.products[i].price

        # user can't click again on products already bought
        prob_matrix[:, user.primary] = 0.

        while round < 5 and np.sum(newly_active_nodes) > 0:

            p = (prob_matrix.T * active_nodes).T
            # take probabilities only for active nodes (rows of active nodes)

            products_clicked = p > np.random.rand(p.shape[0], p.shape[1])
            # matrix with True/False based on products which client would click

            # ((p!=0)==activated_edges) clicked + previously non-zero
            prob_matrix = prob_matrix * ((p != 0) == products_clicked)
            newly_active_nodes = (np.sum(products_clicked, axis=0) > 0) * (1 - active_nodes)

            # user can't click again on product already clicked
            prob_matrix[:, newly_active_nodes == 1] = 0.

            # slots clicked
            secondary_slots = np.where(newly_active_nodes == 1)[0]

            # check idxs which match the reservation price
            stop_idxs = np.where(np.array(user.reservation_price)[secondary_slots] < prod_prices[secondary_slots])
            go_idxs = np.where(np.array(user.reservation_price)[secondary_slots] >= prod_prices[secondary_slots])

            # users don't buy products higher than their reservation price so the visit can't go on for these idxs
            # because no secondary products are shown, since the primary isn't bought
            prob_matrix[secondary_slots[stop_idxs], :] = 0.

            active_nodes = newly_active_nodes
            round += 1

            for i in range(np.size(np.where(newly_active_nodes == 1)[0])):
                history_click.append(np.where(newly_active_nodes == 1)[0][i])
            for i in range(np.size(go_idxs[0])):
                history_purchase.append(secondary_slots[go_idxs[0][i]])
            history_nodes = np.concatenate((history_nodes, [newly_active_nodes]), axis=0)

        user.products_clicked = history_click
        user.cart = history_purchase

        # estimate random quantities for each product
        lambdas = np.array(
            [2, 1, 3, 3, 1])  # these are the mean (poisson) units sold for each product (when it is bought)

        # in the case of fixed units bought (always the same number of units is bought for each product),
        # we take the mean (lambda) of the poisson Step3
        if fixed_units == 1:
            units = lambdas
        # Step4
        else:
            # we don't want zeros (1+) (at least one units is bought)
            units = 1 + np.random.poisson(lambdas - 1, size=(1, 5))[0]

        if len(history_purchase) != 0:
            for i in range(len(history_purchase)):
                user.quantities.append(units[history_purchase[i]])

        return history_nodes
