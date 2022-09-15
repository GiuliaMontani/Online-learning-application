#Greedy Algorithm
#Optimization of the cumulative EXPECTED margin over all the products

def Greedy_algorithm(Conv_matrix, Margins_matrix):
    
    flag = 0
    best_config = [] #to save indexes which icreased their price
    rewards_collected = []
    max_collected = []

    #initial step
    #reward of the configuration 0
    max = sum(Conv_matrix[:,0] * Margins_matrix[:,0])
    print("starting configuration: ", max)
    rewards_collected.append(max)
    max_collected.append(max)
    num_it = 0

    #for each price choice (4)
    for i in range(0,4):
        if flag==1:
            i = 0
        num_it += 1
        #if the iteration doesn't find a better configuration -> stop
        if flag==0 and i!=0:
            break

        #update the margin and the conversion rate of the kth element:
        #if in the previous iteration the maximum was updated it means that the kth product
        #has increased its price by one (since we got a better configuration for the reward)
        #So we don't reset the matrix for it but we set a new price increased by one:
        if flag!=0 and i+1<4:
            Conv_matrix[k,i] = Conv_matrix[k,i+1]
            Margins_matrix[k,i] = Margins_matrix[k,i+1]
            flag = 0
            best_config.append(k)
            max_collected.append(max)
            print(max)
        
        print("Iterazione ", num_it)
        #for each product
        for j in range(0,5):

            #if i+1 is not out of bound -> increase price of jth product by 1
            #and compute reward
            if i+1<4:
                reward = 0
                #we compute the total expected reward as the sum of the rewards of each single product, 
                for a in range(0,5):

                    # except the one which has increased his price
                    if a!=j:
                       reward = reward + Conv_matrix[a,i] * Margins_matrix[a,i]
                #we add the increased one
                reward = reward + Conv_matrix[j,i+1] * Margins_matrix[j,i+1]
            
            #print(reward)
            rewards_collected.append(reward)
            if reward>max:
                max = reward
                k = j #save the product that in this iteration has increased the margin
                flag = 1 #since we foud a new maximum

    print("__________________")
    print("max expected reward found: ", max)
    print("__________________")
    print("best configuration: ")
    for i in range(0, len(best_config)):
      print(i+1, "): ")
      print("increase price of the product", best_config[i]+1)
    return [max,best_config, rewards_collected, max_collected, num_it]