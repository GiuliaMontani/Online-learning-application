import numpy as np
import random


class User:
    # for each class of users, i have an average reservation price
    avg_reservation_price = []
    # constructor
    def __init__(self, primary):
        # for each product a user has a reservation price
        self.reservation_price = []  # 5x1
        # stores the id {0,1,2,3,4} of products clicked
        self.products_clicked = []
        # stores the id {0,1,2,3,4} of products bought
        self.cart = []
        # stores the quantities of items bought for each product
        self.quantities = []
        # graph with the influence probabilities between the products
        self.P = np.zeros((5, 5))
        # primary product shown
        self.primary = primary  # {0,1,2,3,4}


    def generate_graph(self, distribution):
        """ Generate the graph with the click probabilities on the products.

        :param distribution: distribution to be used for the click probabilities(es. np.random.uniform)
        :return: the generated graph.
        """
        graph = np.zeros((5, 5))
        graph = distribution
        return graph

    # Inheritance
# 3 classes of users:

# non-Italian buyers (they are not members)
class User0(User):
    alpha = [0.1, 0.3, 0.1, 0.2, 0.3]
    avg_reservation_price = np.array([23, 34, 31, 46, 104])
    scale = np.array([1.5, 2, 1.7, 2.5, 4])
    @staticmethod
    def reset_avg_reservation_price():
        User0.avg_reservation_price = np.array([23, 34, 31, 46, 104])
        User0.scale = np.array([1.5, 2, 1.7, 2.5, 4])
        print("RESET 0:", User0.avg_reservation_price)

    def __init__(self, primary, fixed_weights):
        User.__init__(self, primary)
        self.f1 = 0
        self.f2 = 0
        self.reservation_price = self.avg_reservation_price + np.random.normal(0, scale=1,
                                                                         size=5)*self.scale
        # uncertain weight of the graph
        if fixed_weights != 1:
            self.P = User.generate_graph(self, np.random.uniform(0, 0.5, size=(5, 5)))  # lower influence probabilities
        # known weight of the graph
        else:
            self.P = User.generate_graph(self,
                                         np.array([[0.1612583, 0.02990957, 0.0215376, 0.27068635, 0.25147687],
                                                   [0.4918155, 0.39001488, 0.12488759, 0.41009003, 0.32833393],
                                                   [0.25435096, 0.32939066, 0.25589946, 0.18488915, 0.36627083],
                                                   [0.27801728, 0.47490127, 0.33253022, 0.48659868, 0.42396207],
                                                   [0.04359364, 0.45397953, 0.31312689, 0.41816953, 0.24363287]]))



# Italian customers who did not subscribe for membership
class User1(User):
    alpha = [0.2, 0.2, 0.2, 0.2, 0.2]
    avg_reservation_price = np.array([21, 32, 29, 44, 95])
    scale = np.array([1.5, 2, 1.7, 2.5, 4])
    @staticmethod
    def reset_avg_reservation_price():
        User1.avg_reservation_price = np.array([21, 32, 29, 44, 95])
        User1.scale = np.array([1.5, 2, 1.7, 2.5, 4])
        print("RESET 1:", User1.avg_reservation_price)

    def __init__(self, primary, fixed_weights):
        User.__init__(self, primary)
        self.f1 = 1
        self.f2 = 0
        self.reservation_price = self.avg_reservation_price + np.random.normal(0, scale=1,
                                                                         size=5)*self.scale  # more variable reservation price
        if fixed_weights != 1:
            self.P = User.generate_graph(self, np.random.uniform(0.2, 1, size=(5, 5)))  # higher influence probabilities
        else:
            self.P = User.generate_graph(self,
                                         np.array([[0.1612583, 0.02990957, 0.0215376, 0.27068635, 0.25147687],
                                                   [0.4918155, 0.39001488, 0.12488759, 0.41009003, 0.32833393],
                                                   [0.25435096, 0.32939066, 0.25589946, 0.18488915, 0.36627083],
                                                   [0.27801728, 0.47490127, 0.33253022, 0.48659868, 0.42396207],
                                                   [0.04359364, 0.45397953, 0.31312689, 0.41816953, 0.24363287]]))


# Italian customers who did subscribe for membership
class User2(User):
    alpha = [0.2, 0.2, 0.2, 0.2, 0.2]
    avg_reservation_price =  np.array([21, 31, 28, 42, 87])
    scale = np.array([1.5, 2, 1.7, 2.5, 3])

    @staticmethod
    def reset_avg_reservation_price():
        User2.avg_reservation_price = np.array([21, 31, 28, 42, 87])
        print("RESET 2:",User2.avg_reservation_price )
        User2.scale = np.array([1.5, 2, 1.7, 2.5, 3])

    def __init__(self, primary, fixed_weights):
        User.__init__(self, primary)
        self.f1 = 1
        self.f2 = 1
        self.reservation_price = self.avg_reservation_price + np.random.normal(0, scale=1, size=5)*self.scale
        if fixed_weights != 1:
            self.P = User.generate_graph(self,
                                         np.random.uniform(0, 1, size=(5, 5)))  # more variable influence probabilities
        else:
            self.P = User.generate_graph(self,
                                         np.array([[0.1612583, 0.02990957, 0.0215376, 0.27068635, 0.25147687],
                                                   [0.4918155, 0.39001488, 0.12488759, 0.41009003, 0.32833393],
                                                   [0.25435096, 0.32939066, 0.25589946, 0.18488915, 0.36627083],
                                                   [0.27801728, 0.47490127, 0.33253022, 0.48659868, 0.42396207],
                                                   [0.04359364, 0.45397953, 0.31312689, 0.41816953, 0.24363287]]))


# homogenous user class (used in the first steps)
class HomogeneousUsers(User):
    alpha = [0.2, 0.2, 0.2, 0.2, 0.2]
    avg_reservation_price = np.multiply([0.3, 0.3, 0.3, 0.3, 0.3],[25, 37, 34, 48, 104]) + np.multiply([0.4, 0.4, 0.4, 0.4, 0.4], [23, 34, 30, 46, 95]) + np.multiply([0.3, 0.3, 0.3, 0.3, 0.3],[21, 31, 28, 42, 87])
    scale = np.array([1.5, 2, 1.7, 2.5, 4])
    def __init__(self, primary, fixed_weights):
        User.__init__(self, primary)
        self.reservation_price = self.avg_reservation_price + np.random.normal(0, scale=1, size=5)*self.scale
        if fixed_weights != 1:
            self.P = User.generate_graph(self, np.random.uniform(0.2, 1, size=(5, 5)))
        else:
            self.P = User.generate_graph(self,
                                         np.array([[0.1612583, 0.02990957, 0.0215376, 0.27068635, 0.25147687],
                                                   [0.4918155, 0.39001488, 0.12488759, 0.41009003, 0.32833393],
                                                   [0.25435096, 0.32939066, 0.25589946, 0.18488915, 0.36627083],
                                                   [0.27801728, 0.47490127, 0.33253022, 0.48659868, 0.42396207],
                                                   [0.04359364, 0.45397953, 0.31312689, 0.41816953, 0.24363287]]))

