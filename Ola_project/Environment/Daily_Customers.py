import numpy as np
from Environment.User import *

# Each day we have a list of users who enter the website, distributed with respect to their classes
class Daily_Customers:
   # constructor
    def __init__(self):
        self.Users = []

    
    # which type of user has to be created (which class)
    def whichUser(self, binary_vector, primary, fixed_weights, binary_features):

      if binary_features==1:  
        
        if np.sum(binary_vector==0):
            self.Users.append(User0(primary,fixed_weights))
        elif np.sum(binary_vector==1):
            self.Users.append(User1(primary,fixed_weights))    
        elif np.sum(binary_vector==2):
            self.Users.append(User2(primary,fixed_weights))
      
      #if the weights are fixed -> users have the same graph with the click probabilities
      else:
          self.Users.append(homogeneous_users(primary, fixed_weights))


    # generate new users for the day
    def UsersGenerator(self, num_users, binary_vector, fixed_alpha, fixed_weights, binary_features):
        #create "alpha_ratio * num_users" users, without the alpha_0 in competitor website
        users_per_product = np.random.multinomial(num_users, np.random.dirichlet(np.ones(5)))
        #if alpha are not uncertain. we suppose users are equally distributed
        if fixed_alpha == 1:
            users_per_product = np.ones(5) * round(num_users/5)
        for i in range(len(users_per_product)):
            for j in range(int(users_per_product[i])):
                self.whichUser(binary_vector, i,fixed_weights, binary_features) #i is the index of the primary product
                
                