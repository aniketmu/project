# libraries
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random as rnd
import pandas as pd
# end



# function to calculate spectral radius and theta steps
# returns None, plots
def calculate(T=1,steps = 10,alpha=0.2,c=1):
    
    # slope calculation
    slope = (1-0)/(-((5-(T/10))*T)) # it was (0.1 - 5)T

    # creating list of theta_steps
    theta_step = [(0.1*i)*T for i in range(1,steps)]


    #Q_0 = 50
    #T = Q_0 * np.exp(rnd.uniform(0, 1)*slope)
    
    
    # blank list of radius for each theta step
    list_rads = []
    for each_theta_step in theta_step:
        

        weight_matrix = []
        for i in range(1,11):

            # blank row
            row = []
            for j in range(1,11):

                # assigning values to matrix

                # for diagonal
                if i==j:
                    row.append(alpha)

                # for non-diagonal
                else:

                    # calculating time differance
                    time_diff = (j-i)*each_theta_step

                    # appending slope to the matrix
                    row.append(slope*(time_diff-(0.1*T))+c)


            # appending row to the blank matrix
            weight_matrix.append(row)

        #print(len(weight_matrix),len(weight_matrix[0]))

        # to look the matrix as dataframe
        #print(pd.DataFrame(weight_matrix))

        # finding max radius of the spectrum
        radius = abs(max(LA.eigvals(weight_matrix)))
        list_rads.append(radius)
    
    # plotting part
    plt.plot(theta_step,list_rads)
    plt.xlabel('Theta_steps')
    plt.ylabel('Spectral_radius')
    plt.savefig("./figures/"+str(T)+".jpg")
    plt.show()
    #plt.clf()
# end of the function    


if __name__ == '__main__':

    for i in range(1,11):

        calculate(i)



