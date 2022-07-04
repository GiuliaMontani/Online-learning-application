import numpy as np
from Environment.User import *

# Each day we have a list of users who enter the website, distributed with respect to their classes
class Daily_Customers:
   # constructor
    def __init__(self):
        self.Users = []

    
    # which type of user has to be created (which class)
    def whichUser(self, binary_vector, primary,fixed_weights):
      if fixed_weights!=1:  
        if np.sum(binary_vector==0):
            self.Users.append(User0(primary))
        elif np.sum(binary_vector==1):
            self.Users.append(User1(primary))    
        elif np.sum(binary_vector==2):
            self.Users.append(User2(primary))
      else:
          self.Users.append(FixedUsers(primary))


    # generate new users for the day
    def UsersGenerator(self, num_users, binary_vector, fixed_alpha, fixed_weights):
        #create "alpha_ratio * num_users" users, without the alpha_0 in competitor website
        users_per_product = np.random.multinomial(num_users, np.random.dirichlet(np.ones(5)))
        if fixed_alpha == 1:
            users_per_product = np.ones(5) * round(num_users/5)
        for i in range(len(users_per_product)):
            for j in range(int(users_per_product[i])):
                self.whichUser(binary_vector, i,fixed_weights) #i is the index of the primary product
                
                