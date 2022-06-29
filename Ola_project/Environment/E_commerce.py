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
        #number of clicks per product
        self.graph = self.generate_graph(np.ones(5))
        #list of the total daily rewards for each day
        self.daily_rewards = []
        #clicks per product for the current day
        self.daily_clicks = []
        self.daily_purchases = []


    def set_lambda(self, new_lambda):
        self.lambda_ = new_lambda
    
    
    def add_product(self, product):
        self.products.append(product)
    
    
    def set_products(self, product_list):
        self.products = product_list


    def clear_history(self):
        self.time_history[:] = []
        self.daily_rewards[:] = []
        self.daily_purchases[:] = []
        self.daily_clicks[:] =  []
        self.daily_users[:] = []



    # graph with the probabilities to see the products 
    # 1 for the first secondary slot, lambda for the second one
    def generate_graph(self, distribution):
        graph = np.zeros((5,5))
        for i in range(5):
          # secondary slots indexes (0,1,2,3,4,5)-{i=primary}
          j = np.random.choice([x for x in range(5) if x != i ],2, replace = False) 
          # probability to see the first slot = 1
          graph[i,j[0]] = distribution[i]
          # 1 * lambda 
          graph[i,j[1]] = distribution[i] * self.lambda_
        return graph
    

    # simulate a day of visits in the website
    def simulate_day(self, num_users, binary_vector, fixed_alpha):
        D = Daily_Customers()
        D.UsersGenerator(num_users, binary_vector, fixed_alpha)
        self.daily_users.append(D.Users)
        rewards_of_the_day = 0
        # store the visits of the day
        Day = []
        clicks_per_product = np.array([0,0,0,0,0])
        purchases_per_product = np.array([0,0,0,0,0])
        # for each user visit (each day we can change the prices: we have to implement it)
        for i in range(num_users):
            visit = self.visit(D.Users[i])
            Day.append(visit)
            #for each user compute reward from the cart (sum of the margins of the products bought)
            #and update purchase count for each product bought
            for k in range(np.size(D.Users[i].cart)):          
                rewards_of_the_day += self.products[D.Users[i].cart[k]].margin
                purchases_per_product[D.Users[i].cart[k]]+= 1
            #for each user update clicks count
            for z in range(np.size(D.Users[i].products_clicked)):
                clicks_per_product[D.Users[i].products_clicked[z]]+= 1

        self.time_history.append(Day)
        self.daily_rewards.append(rewards_of_the_day)
        self.daily_purchases=purchases_per_product
        self.daily_clicks= clicks_per_product
        D.Users[:] = []
        return Day
      


    # simulate when an user visits the website
    def visit(self, user):
        # Influence probability matrix of the products, for each user equal to the see probability*click probability
        prob_matrix = user.P * self.graph
        n_nodes = prob_matrix.shape[0]

        # if user's reservation price is lower than the price of the primary product -> end the visit
        if user.reservation_price[user.primary] < self.products[user.primary].price:
            user.products_clicked = [user.primary]
            history_purchase = []
            user.cart = []
            user.quantities = np.zeros(user.P.shape[0])
            active_nodes = np.zeros(n_nodes)
            active_nodes[user.primary] = 1
            return np.array([active_nodes])

        # Influence probability matrix of the products, for each user equal to the see probability*click probability
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

        #user can't click again on products already bought
        prob_matrix[:, user.primary] = 0.

        while(round < 5 and np.sum(newly_active_nodes)>0):

            p = (prob_matrix.T * active_nodes).T
            products_clicked = p > np.random.rand(p.shape[0], p.shape[1])

            prob_matrix = prob_matrix * ((p!=0)==products_clicked)
            newly_active_nodes = (np.sum(products_clicked, axis=0)>0) * (1-active_nodes)

            #user can't click again on product already clicked
            prob_matrix[:, newly_active_nodes==1] = 0.
            
            # slots clicked
            secondary_slots = np.where(newly_active_nodes == 1)[0] 

            # check idxs which match the reservation price
            stop_idxs = np.where(np.array(user.reservation_price)[secondary_slots] < prod_prices[secondary_slots])
            go_idxs = np.where(np.array(user.reservation_price)[secondary_slots] >= prod_prices[secondary_slots])

            #users don't buy products higher than their reservation price
            #so the visit can't go on for theese idxs because no secondary products are shown, since the primary isn't bought
            prob_matrix[secondary_slots[stop_idxs],:] = 0.

            active_nodes = newly_active_nodes
            round += 1

            for i in range(np.size(np.where(newly_active_nodes==1)[0])):
                history_click.append(np.where(newly_active_nodes==1)[0][i])
            for i in range(np.size(go_idxs[0])):
                history_purchase.append(secondary_slots[go_idxs[0][i]])
            history_nodes = np.concatenate((history_nodes, [newly_active_nodes]),axis=0)

        user.products_clicked = history_click
        user.cart = history_purchase

        # estimate random quantities for each product
        if(len(history_purchase)!=0):
            for i in range(len(history_purchase)):
                if(np.size(history_purchase[i])!=0):
                    for j in range(np.size(history_purchase[i])):
                        user.quantities.append(1 + np.random.randint(3, size = 1)[0]) # from 1 to 3 units bought

        return history_nodes
    