# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 18:39:13 2017

@author: mj
"""
import numpy as np
import time 

N = 500
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.rand.html

#2.1
def forever():
    print('\n\n2.1')
    matrix1 = np.random.rand(N,N)
    matrix2 = np.random.rand(N,N)
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
           for k in range(len(matrix2)):
               matrix1[i][k] * matrix2[k][j]
  

startingTime = time.clock()             
forever();
endingTime = time.clock()
foreverTime = endingTime - startingTime
print ("Time elpased to run forever: {0}".format(foreverTime))


#2.2
def mat_fast():
    print('\n\n2.2')
    startingTime = time.clock()
    matrix1 = np.random.rand(N,N)
    matrix2 = np.random.rand(N,N)
    matrix1.dot(matrix2)
    endingTime = time.clock()
    mat_fastTime = endingTime - startingTime
    print ("Time elpased to run mat_fast : {0}".format(mat_fastTime))
    
startingTime = time.clock()                 
mat_fast();
endingTime = time.clock()
mat_fastTime = endingTime - startingTime

#2.3
print('\n\n2.3')
print ("Time Difference : {0}".format(foreverTime-mat_fastTime))
