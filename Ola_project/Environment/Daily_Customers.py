import numpy as np
from Environment.User import *


# Each day we have a list of users who enter the website, distributed with respect to their classes
class Daily_Customers:
    # constructor
    def __init__(self):
        self.Users = []

    def whichUser(self, binary_vector, primary, fixed_weights, binary_features):
        """add a User to the Daily Customers based on its type

        :param binary_vector: type of user (0,1 or 2)
        :param primary: primary product which is shown
        :param fixed_weights: 1 if alpha is fixed
        :param binary_features: 1 if we do not distinguish between users (STEP 1)

        """

        if binary_features == 1:

            if np.sum(binary_vector == 0):
                self.Users.append(User0(primary, fixed_weights))
            elif np.sum(binary_vector == 1):
                self.Users.append(User1(primary, fixed_weights))
            elif np.sum(binary_vector == 2):
                self.Users.append(User2(primary, fixed_weights))

        # if the weights are fixed -> users have the same graph with the click probabilities
        else:
            self.Users.append(homogeneous_users(primary, fixed_weights))

    def UsersGenerator(self, num_users, binary_vector, fixed_alpha, fixed_weights, binary_features, alpha=np.ones(5)):
        """Generate daily users choosing which product they see first (if they arrive at the website) based on their
        type.

        :param num_users: average number of potential users in a day
        :type num_users: int
        :param binary_vector: type of user (0,1 or 2)
        :type binary_vector: int
        :param fixed_alpha: 1 if alpha is fixed (uniformly distributed over the products)
        :type fixed_alpha: bool
        :param fixed_weights: 1 if alpha is fixed
        :type fixed_weights: bool
        :param binary_features: 1 if we do not distinguish between users (STEP 1)
        :type binary_features: bool
        :param alpha: vector with dirichlet parameters, by default each

        """
        # create "alpha_ratio * num_users" users, without the alpha_0 in competitor website
        users_per_product = np.random.multinomial(num_users, np.random.dirichlet(alpha))
        # <--- QUI NON STIAMO CONSIDERANDO QUELLI CHE VANNO DAI CONCORRENTI

        # if alpha are not uncertain. we suppose users are equally distributed
        if fixed_alpha == 1:
            users_per_product = np.ones(5) * round(num_users / 5)
        for i in range(len(users_per_product)):
            for j in range(int(users_per_product[i])):
                self.whichUser(binary_vector, i, fixed_weights,
                               binary_features)  # i is the index of the primary product
