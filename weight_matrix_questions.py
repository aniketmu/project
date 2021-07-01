#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random as rnd
import pandas as pd


# In[6]:


def calculate(T=1,steps = 10, alpha=0.2, c=1):
    slope = (1-0)/(0.1*T-0.5*T)   #doubt in the denominator
    theta_step = [(0.1*i)*T for i in range(1,steps)]
    
    list_spectralrad = []
    for each_theta_step in theta_step:
        
        weight_matrix = []
        for i in range(1,11):
            
            row = []
            for j in range(1,11):
                
                if i==j:
                    row.append(alpha)
                else:
                         time_diff = (j-i)*each_theta_step
                         row.append(slope*(time_diff-(0.1*T))+c)
                        
            weight_matrix.append(row)
            
            #print(len(weight_matrix),len(weight_matrix[0]))
            print(pd.DataFrame(weight_matrix))
            
            spectral_radius = abs(max(LA.eigvals(weight_matrix)))
            list_spectralrad.append(spectral_radius)
            
            plt.plot(theta_step,list_rads)
            plt.xlabel('Theta_steps')
            plt.ylabel('Spectral_radius')
            plt.show()

