class Product:
    # constructor
    def __init__(self, id_, price_list, margins_list):
        self.id_ = id_
        # possible choices of price
        self.price_list = price_list
        # actual price
        self.price = price_list[0]
        # index of the actual price
        self.idx = 0
        #margin for each price
        self.margins_list = margins_list
        self.margin = margins_list[0]

    # set the four new possible choices of the prices
    def set_new_price_list(self, price_list):
        self.price_list = price_list
        self.price = price_list[0]


    # change the actual price between one of the possible choice ordered with increasing prices
    def change_price(self, new_index):
        self.price = self.price_list[new_index]
        self.margin = self.margins_list[new_index]

    # increase by one step the price in the vector of possible prices
    def increase_price(self):
        if self.price == max(self.price_list):
            return
        else:
            self.price = self.price_list[self.idx + 1]
            self.margin = self.margins_list[self.idx + 1]
            self.idx += 1


    # decrease by one step the price in the vector of possible prices
    def decrease_price(self):
        if self.price == min(self.price_list):
            return
        else:
            self.price = self.price_list[self.idx - 1]
            self.margin = self.margins_list[self.idx - 1]
            self.idx -= 1