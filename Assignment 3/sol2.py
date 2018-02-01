# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 01:36:14 2017

@author: mj
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# 2.1
print("\n\n2.1")
dice_vector = np.zeros(20)
for i in range(0, 20):
    dice_vector[i] = np.random.randint(1,7) + np.random.randint(1,7)
    
hist = plt.hist(dice_vector)   
plt.show() 

for i in hist[0]:
    print("Probability is {0}".format(i/hist[0].sum()))

hist = np.histogram(dice_vector)

# 2.2
print("\n\n2.2")
dice_vector = np.zeros(1000)
for i in range(0, 1000):
    dice_vector[i] = np.random.randint(1,7) + np.random.randint(1,7)
hist = plt.hist(dice_vector)
plt.show()   
for i in hist[0]:
    print("Probability is {0}".format(i/hist[0].sum()))
    
# 2.3
print("\n\n2.3")
dice_vector = np.zeros(1000)
for i in range(0, 1000):
    dice_vector[i] = np.random.randint(1,7) + np.random.choice([1,0,3,4,5,6])
    
hist = plt.hist(dice_vector)
plt.show()
for i in hist[0]:
    print("Probability is {0}".format(i/hist[0].sum()))

# 2.4
print("\n\n2.4")
dice_vector = np.zeros(1000)
for i in range(0, 1000):
    dice_vector[i] = (np.random.randint(1,7) + np.random.randint(1,7) +
                np.random.randint(1,7) + np.random.randint(1,7) +
                np.random.randint(1,7) + np.random.randint(1,7) +
                np.random.randint(1,7))
    
plt.hist(dice_vector, normed=True)
x = np.linspace(min(dice_vector), max(dice_vector), 100)
plt.plot(x, mlab.normpdf(x, np.mean(dice_vector),
                         np.sqrt(np.var(dice_vector))), color = 'y')
plt.show()
print("Mean {0}".format(dice_vector.mean()))
print("Variance {0}".format(dice_vector.var()))