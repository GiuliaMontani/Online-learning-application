# OLA_Project

This Online Learning Applications project was developed by Aygalic Jara, Christopher Volpi, Giulia Montani, Luca Mainini and Roberta Troilo. 
It aimed to implement a pricing algorithm based on multi-armed bandit for a small e-commerce platform.

## E-COMMERCE

<img width="947" alt="image" src="https://user-images.githubusercontent.com/57671317/191115356-11d6c612-28b9-497f-affe-9caea15cda44.png">

In every webpage, a single product, called primary, is displayed together with its price. The user can add a number of units of this product to the cart. After the product has been added to the cart, two products, called secondary, are recommended. When displaying the secondary products, the price is hidden. Furthermore, the products are recommended in two slots, one above the other, thus providing more importance to the product displayed in the slot above. If the user clicks on a secondary product, a new tab on the browser is opened and, in the loaded webpage, the clicked product is displayed as primary together with its price. At the end of the visit over the ecommerce website, the user buys the products added to the cart.

<img width="947" alt="image" src="https://user-images.githubusercontent.com/57671317/191115409-d2bc5652-4e54-498c-b5d6-8d5241238144.png">

In our simulation, we consider 3 classes of users that had different reservation prices:
<img width="960" alt="image" src="https://user-images.githubusercontent.com/57671317/194412744-7594d4d9-3d9f-4b7a-82b5-71441ef6be9b.png">
<img width="693" alt="image" src="https://user-images.githubusercontent.com/57671317/194412418-ee43950c-465e-4a68-bc0b-de5085c51cdc.png">


## ENVIRONMENT
We implemented different classes for the simulation:
- User
- Product
- Daily Customer
- Ecommerce

## IMPLEMENTED ALGORITHMS
- Greedy algorithm
- UCB 
- Sliding UCB
- UCB with CUSUM (change detectors)
- Thompson Sampling
- Dynamic UCB and TS



