import numpy as np
from Environment.User import *


# Each day we have a list of users who enter the website, distributed with respect to their classes
class Daily_Customers:
    # constructor
    def __init__(self, percentage=[0.3,0.4,0.3]):
        self.Users = []
        self.users_distribution = percentage

    def whichUser(self, type_user, primary, fixed_weights, binary_features):
        """add a User to the Daily Customers based on its type

        :param type_user: type of user (0,1 or 2)
        :param primary: primary product which is shown
        :param fixed_weights: 1 if alpha is fixed
        :param binary_features: 1 if we do not distinguish between users (STEP 1)

        """

        if binary_features == 1:

            if type_user == 0:
                self.Users.append(User0(primary, fixed_weights))
            elif type_user == 1:
                self.Users.append(User1(primary, fixed_weights))
            elif type_user == 2:
                self.Users.append(User2(primary, fixed_weights))

        # if the weights are fixed -> users have the same graph with the click probabilities
        else:
            self.Users.append(HomogeneousUsers(primary, fixed_weights))

    def UsersGenerator(self, number_users, fixed_alpha, fixed_weights, binary_features):
        """Generate daily users choosing which product they see first (if they arrive at the website) based on their
        type.

        :param number_users: average number of potential users in a day
        :type number_users: int
        :param fixed_alpha: 1 if alpha is fixed (uniformly distributed over the products)
        :type fixed_alpha: bool
        :param fixed_weights: 1 if alpha is fixed
        :type fixed_weights: bool
        :param binary_features: 1 if we distinguish between user's types, 0 if not
        :type binary_features: bool

        """

        # if alpha are not uncertain, we suppose users are equally distributed (ASSUMPTION)
        if fixed_alpha == 1 and binary_features == 0:
            num_users = int(np.random.normal(number_users, scale=0.2 * number_users, size=1))  # drawn from a gaussian
            users_per_product = np.ones(5) * round(num_users / 5)

        if binary_features == 1:
            for type_user in range(3):

                if fixed_alpha == 1:
                    num_users = int(
                        np.random.normal(number_users*self.users_distribution[type_user], scale=0.2 * number_users*self.users_distribution[type_user], size=1))  # drawn from a gaussian
                    users_per_product = np.ones(5) * round(num_users / 5)
                    # users_per_product = np.random.multinomial(num_users, alpha)
                else:  # fixed_alpha == 0
                    num_users = int(np.random.normal(number_users*self.users_distribution[type_user], scale=0.2 * number_users*self.users_distribution[type_user], size=1))  # drawn from a gaussian

                    if type_user == 0:
                        alpha = User0.alpha
                    elif type_user == 1:
                        alpha = User1.alpha
                    elif type_user == 2:
                        alpha = User2.alpha
                    #  media = np.random.dirichlet(alpha)  #alpha ha 6 elem
                    #  users_per_product = np.random.multinomial(num_users, np.random.dirichlet(alpha[1:6]))
                    users_per_product = np.random.multinomial(num_users, np.random.dirichlet(alpha))  # ho tolto alpha0
                    # create "alpha_ratio * num_users" users, without the alpha_0 in competitor website
                    # <--- QUI NON STIAMO CONSIDERANDO QUELLI CHE VANNO DAI CONCORRENTI

                for i in range(len(users_per_product)):  # for each product
                    # for j in range(int(users_per_product[i])):  # for each user
                    self.whichUser(type_user, i, fixed_weights,
                                   binary_features)  # i is the index of the primary product

        if fixed_alpha == 0 and binary_features == 0:
            alpha = HomogeneousUsers.alpha
            num_users = int(np.random.normal(number_users, scale=0.2 * number_users,
                                             size=1))  # drawn from a gaussian
            users_per_product = np.random.multinomial(num_users, np.random.dirichlet(alpha))
            for i in range(len(users_per_product)):  # for each product
                # for j in range(int(users_per_product[i])):  # for each user
                self.whichUser(-1, i, fixed_weights,
                               binary_features)  # i is the index of the primary product

