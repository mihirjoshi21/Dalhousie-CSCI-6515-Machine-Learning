# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 13:39:38 2017

@author: mj
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools as it

# 1.1
iris_data = np.loadtxt('fisheriris.data', delimiter=',')
feature_names = np.genfromtxt('attributes.txt', delimiter=',', dtype='str').tolist()
target_names = np.genfromtxt('feature.txt', delimiter=',', dtype='str')   
    
# 1.2
print('\n\nAns 1.2')
target = iris_data.T[-1]    
target.shape = (1,150)
iris_data = np.delete(iris_data, np.s_[4::1], 1)
feature_names.pop(4)
print ("size of the Fisher's measurements: {0}".format(np.size(iris_data)))
print ("number of elements of the second dimension : {0}".format(len(iris_data.T[1])))

  
# 1.3
print('\n\nAns 1.3')
print ("sums of each of the columns in the irirs_data : {0}"
       .format(iris_data.sum(axis=0)))

print ("sums of just the second and the fourth columns: {0}"
       .format(iris_data[:, [1, 3]].sum(axis=0)))

print ("maximum value from 27 through 48 in col 1: {0}"
       .format((float)(np.max(iris_data[26:48,0:1]))))
print ("maximum value from 27 through 48 in col 2: {0}"
       .format((float)(np.max(iris_data[26:48,1:2]))))
print ("maximum value from 27 through 48 in col 3: {0}"
       .format((float)(np.max(iris_data[26:48,2:3]))))
print ("maximum value from 27 through 48 in col 4: {0}"
       .format((float)(np.max(iris_data[26:48,3:4]))))

print ("minimum value from odd numbered rows 3rd and 33rd rows in col 1: {0}"
       .format((float)(np.min(iris_data[3:33:2,0:1]))))
print ("minimum value from odd numbered rows 3rd and 33rd rows in col 2: {0}"
       .format((float)(np.min(iris_data[3:33:2,1:2]))))


# 1.4
print('\n\nAns 1.4')
r13rd = iris_data.T[0] + iris_data.T[2]
print ("r13rd sum: {0}"
       .format(r13rd.sum()))
print ("r13rd cube: {0}"
       .format(np.power(r13rd, 3)))


# 1.5
print('\n\nAns 1.5')
mat1 = iris_data[0:4,0:2]
mat2 = iris_data[0:4,2:4]
print ("addition of mat1 and mat2: \n{0}"
       .format(np.add(mat1, mat2)))
print ("muliplication of mat1 and mat2: \n{0}"
       .format(np.multiply(mat1, mat2)))


# 1.6
print('\n\nAns 1.6')
mat3 = np.inner(mat1,mat2)
print ("inner product of mat1 and mat2: \n{0}"
       .format(mat3))
np.savetxt('mat3.csv', mat3, delimiter=',', fmt='%.2f')

# 1.7
print('\n\nAns 1.7')

mean = [np.mean(iris_data.T[0]).mean(), np.mean(iris_data.T[1]).mean(),
        np.mean(iris_data.T[2]).mean(), np.mean(iris_data.T[3]).mean()]

sd = [np.std(iris_data.T[0]).mean(), np.std(iris_data.T[1]).mean(),
        np.std(iris_data.T[2]).mean(), np.std(iris_data.T[3]).mean()]

x_axis = list(range(len(feature_names)))

fig = plt.figure(figsize=(10, 5))
plt.bar(x_axis, mean, yerr=sd)
        
plt.grid()
y_axis = max(zip(mean, sd))
plt.ylim([0, (y_axis[0] + y_axis[1])])
plt.title('Mean and standard deviation for Iris Data')
plt.ylabel('Mean length in cm')
plt.xticks(x_axis, feature_names)
plt.show()


# 1.8
print('\n\nAns 1.8')
markers = ['o', 'v', '*']
colors = ['y', 'b', 'r']

fig = plt.figure(figsize=(10, 5))

setosa = plt.scatter(np.transpose(iris_data[0:50, [0, 1]])[0],
            np.transpose(iris_data[0:50, [0, 1]])[1], marker=markers[0], 
            color=colors[0])
versicolor = plt.scatter(np.transpose(iris_data[50:100, [0, 1]])[0], 
            np.transpose(iris_data[50:100, [0, 1]])[1], marker=markers[1], color=colors[1])
virginica = plt.scatter(np.transpose(iris_data[100:150, [0, 1]])[0], 
            np.transpose(iris_data[100:150, [0, 1]])[1], marker=markers[2], color=colors[2])

plt.title('Scatter plot of sepal length vs sepal width')
plt.ylabel('sepal width in cm')
plt.xlabel('sepal length in cm')
plt.legend((setosa, versicolor, virginica),
           ('Setosa', 'Versicolor', 'virginica'))
plt.show();


# 1.9
def plotScatter(fig, plotnumber, x, y, label):
     plotnumber = 231 + plotnumber
     subplot = fig.add_subplot(plotnumber) 
     setosa = subplot.scatter(x[0:50],y[0:50],
                              marker=markers[0], color=colors[0])
     versicolor = subplot.scatter(x[50:100],y[50:100], marker=markers[1],
                                    color=colors[1])
     virginica = subplot.scatter(x[100:150],y[100:150],
                                 marker=markers[2], color=colors[2])
     subplot.set_title("Scatter plot of "+label[0] +" vs " + label[1])
     subplot.set_xlabel(label[0])
     subplot.set_ylabel(label[1])
     subplot.legend((setosa, versicolor, virginica),
          ('Setosa', 'Versicolor', 'virginica'), loc=2)
     return subplot

print('\n\nAns 1.9')
fig = plt.figure(figsize=(20, 6))
plotnumbercount=0;
for data, label in zip(list(it.combinations(iris_data.T, 2)),
                               list(it.combinations(feature_names, 2))):
    plotScatter(fig, plotnumbercount, data[0], data[1], label)
    plotnumbercount+=1
plt.tight_layout()
plt.show()


#1.10
print('\n\nAns 1.10')

cent = np.random.randn(5,5)
print("Value of cent: {0}".format(cent))
rs = np.random.RandomState(8)
cent_fix = rs.randn(5,5);
print("Value of cent_fix: {0}".format(cent))

np.random.seed(3)
cent = np.random.randn(5,5)
rs.seed(3)
cent_fix = rs.randn(5,5);
print("Value of cent using seed 3: \n{0}".format(cent))
print("Value of cent_fix using seed 3: \n{0}".format(cent))

np.random.seed(4294967295)
cent = np.random.randn(5,5)
rs.seed(4294967295)
cent_fix = rs.randn(5,5);
print("Value of cent using seed 4294967295: \n{0}".format(cent))
print("Value of cent_fix using seed 4294967295: \n{0}".format(cent))

# 1.11
print('\n\nAns 1.11')
v = np.random.rand(5)
v = np.append(v, np.random.rand(5))
print("Value of v: {0}".format(v))

