# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:24:36 2022

@author: giuli
"""

# STEP 4
# Q1: Il numero di items venduti è una cosa a se? Non influenza nulla sull'algoritmo precedente?
#     Mi semrba che possa rientrare solo nella formula del reward
# Q2: Per gli alpha potremmo usare una Poisson oppure una Gamma

# The Poisson distribution is a discrete distribution that measures the probability of 
# a given number of events happening in a specified time period. 
# In finance, the Poisson distribution could be used to model the arrival of new buy or sell orders 
# entered into the market or the expected arrival of orders at specified trading venues or dark pools. 
# In these cases, the Poisson distribution is used to provide expectations surrounding confidence 
# bounds around the expected order arrival rates. Poisson distributions are very useful for smart order
# routers and algorithmic trading

# In generale, la probabilità che un singolo evento si realizzi dopo un tempo t segue una distribuzione
# esponenziale negativa[2] di parametro λ, ovvero il tempo medio di attesa; la probabilità che n eventi
# si verifichino dopo il tempo t è la somma di tali esponenziali, ed è quindi una V.C. di Earlang[3], 
# caso particolare di una Gamma[4], con parametri n (intero) e λ.

# Update : Gamma (α+ny,β+n)
# Gamma is the cnjugate posteriori of the priori Poisson

# Poisson per numero di items venduti e Gamma per alpha ratio? 
# Non mi convince la Gamma perchè ha il dominio su [0,inf] invece a noi servirebbe qualcosa su (0,1)