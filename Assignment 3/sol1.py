# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:38:11 2017

@author: mj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1.1
df = pd.read_csv('houses.csv')
print(df.describe())

X = df.iloc[:,1:]
Y = df.iloc[:,0]
X_1 = pd.DataFrame(df.sqft_living)

df_norm = (df - df.mean()) / (df.max() - df.min())


def predict(X_1, Y, params):
    
    plt.scatter(X_1, Y,  color='black')
    plt.plot(X_1, params, color='blue', linewidth=1)
    plt.xticks(())
    plt.yticks(())
    plt.show()  

def linear_regression(X, y, theta, alpha, iterations):
    
    X = np.hstack((X,np.ones((X.shape[0],1))))
    y = np.array(Y).flatten()
    m = len(y) 
    cost_history = [0] * iterations
    for i in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis-y
        grad = X.T.dot(loss)/X.size
        theta = theta - alpha*grad
        cost_history[i] = (1.0 / (2 * m)) * np.sum((X.dot(theta)-y)**2)
        
    return theta, cost_history, X, y


def linear_regression_Ridge(X, y, theta, alpha, lam, iterations):
    
    X = np.hstack((X,np.ones((X.shape[0],1))))
    y = np.array(Y).flatten()
    m = len(y) 
    cost_history = [0] * iterations
    for i in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis-y
        theta = theta - (alpha / X.size) * \
                (np.dot(X.T, loss) + lam * theta)
        cost_history[i] = (1.0 / (2 * m)) * \
            (np.sum((np.dot(X, theta) - y) ** 2.) + lam * np.dot(theta.T, theta))
        
    return theta, cost_history, X, y


def linear_regression_LASSO(X, y, theta, alpha, lam, iterations):
    
    X = np.hstack((X,np.ones((X.shape[0],1))))
    y = np.array(Y).flatten()
    m = len(y) 
    cost_history = [0] * iterations
    for i in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis-y
        theta = theta - (alpha / X.size) * \
                (np.dot(X.T, loss) + lam)
        cost_history[i] = (1.0 / (2 * m)) * \
            (np.sum((np.dot(X, theta) - y) ** 2.) + lam * np.sum(theta))
        
    return theta, cost_history, X, y



def linear_regression_LASSO_momentum(X, y, theta, alpha, lam, momenta, iterations):
    
    X = np.hstack((X,np.ones((X.shape[0],1))))
    y = np.array(Y).flatten()
    m = len(y) 
    cost_history = [0] * iterations
    for i in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis-y
        theta = theta - (alpha / X.size) * \
                (np.dot(X.T, loss) + lam) + momenta*theta[len(theta)-1]
        cost_history[i] = (1.0 / (2 * m)) * \
            (np.sum((np.dot(X, theta) - y) ** 2.) + lam * np.sum(theta))
        
    return theta, cost_history, X, y



# 1.2
learning_rates = {10 , 1.0, 0.1, 0.01, 0.001}
for learning_rate in learning_rates:
    theta = np.zeros((2))
    theta, cost_history, train_X, train_y = (
            linear_regression(pd.DataFrame(df_norm.sqft_living),
                     df_norm.iloc[:,0], theta, learning_rate, 10000))
    predict_Y_for_X1 = train_X.dot(theta)
    predict(train_X[:,0], train_y, predict_Y_for_X1)
    
    print("Mean square error for X_1 using linear regression {0}".format(
            ((predict_Y_for_X1 - train_y) ** 2).mean()))
    
    theta = np.zeros((len(df_norm.columns)))
    theta, cost_history_all, train_X, train_y = (
            linear_regression(df_norm.drop('price (grands)', axis=1),
                     df_norm.iloc[:,0], theta, learning_rate, 10000))
    predict_Y = train_X.dot(theta)
    #predict(train_X[:,2], train_y, predict_Y)
    print("Mean square error for X using linear regression {0}".format(
            ((predict_Y - train_y) ** 2).mean()))
    
    plt.yscale('log')
    plt.plot(cost_history)
    plt.plot(cost_history_all, color='yellow')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Function')
    plt.title('Learning rate - {0}'.format(learning_rate))
    plt.show()

    
# 1.3
print("\n\n1.3")
theta = np.zeros((2))
theta, cost_history, train_X, train_y = (
        linear_regression(pd.DataFrame(df_norm.sqft_living),
                          df_norm.iloc[:,0], theta, 1, 10000))
predict_Y_for_X1 = train_X.dot(theta)
predict(train_X[:,0], train_y, predict_Y_for_X1)


theta = np.zeros((len(df_norm.columns)))
theta, cost_history_all, train_X, train_y = (
        linear_regression(df_norm.drop('price (grands)', axis=1),
                          df_norm.iloc[:,0], theta, 1, 10000))
predict_Y = train_X.dot(theta)
plt.scatter(predict_Y, train_y, color='blue', linewidth=1)
plt.show()

#1.4
penalties = {0.1, 0.01}
print("\n\n1.4 For ridge")
for penalty in penalties:
    theta = np.zeros((2))
    theta, cost_history, train_X, train_y = (
            linear_regression_Ridge(pd.DataFrame(df_norm.sqft_living),
                     df_norm.iloc[:,0], theta, 1, penalty, 10000))
    predict_Y_for_X1 = train_X.dot(theta)
    predict(train_X[:,0], train_y, predict_Y_for_X1)
    print("Mean square error for X_1 using ridge {0}".format(
            ((predict_Y_for_X1 - train_y) ** 2).mean()))
    
    theta = np.zeros((len(df_norm.columns)))
    theta, cost_history_all, train_X, train_y = (
            linear_regression_Ridge(df_norm.drop('price (grands)', axis=1),
                     df_norm.iloc[:,0], theta, 1, penalty,  10000))
    predict_Y = train_X.dot(theta)
    print("Mean square error for X using ridge {0}".format(
            ((predict_Y - train_y) ** 2).mean()))
    
    plt.yscale('log')
    plt.plot(cost_history)
    plt.plot(cost_history_all, color='yellow')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Function')
    plt.title('Learning rate - {0}'.format(learning_rate))
    plt.show()


theta = np.zeros((2))
theta, cost_history, train_X, train_y = (
        linear_regression_Ridge(pd.DataFrame(df_norm.sqft_living),
                          df_norm.iloc[:,0], theta, 1, .01, 10000))
predict_Y_for_X1 = train_X.dot(theta)
predict(train_X[:,0], train_y, predict_Y_for_X1)


theta = np.zeros((len(df_norm.columns)))
theta, cost_history_all, train_X, train_y = (
        linear_regression_Ridge(df_norm.drop('price (grands)', axis=1),
                          df_norm.iloc[:,0], theta, 1, .01, 10000))
predict_Y = train_X.dot(theta)
plt.scatter(predict_Y, train_y, color='blue', linewidth=1)
plt.show()



print("\n\n1.4 For lasso")
for penalty in penalties:
    theta = np.zeros((2))
    theta, cost_history, train_X, train_y = (
            linear_regression_LASSO(pd.DataFrame(df_norm.sqft_living),
                     df_norm.iloc[:,0], theta, 1, penalty, 10000))
    predict_Y_for_X1 = train_X.dot(theta)
    predict(train_X[:,0], train_y, predict_Y_for_X1)
    print("Mean square error for X_1 using lasso {0}".format(
            ((predict_Y_for_X1 - train_y) ** 2).mean()))
    
    theta = np.zeros((len(df_norm.columns)))
    theta, cost_history_all, train_X, train_y = (
            linear_regression_LASSO(df_norm.drop('price (grands)', axis=1),
                     df_norm.iloc[:,0], theta, 1, penalty,  10000))
    predict_Y = train_X.dot(theta)
    print("Mean square error for X using lasso {0}".format(
            ((predict_Y - train_y) ** 2).mean()))
    
    plt.yscale('log')
    plt.plot(cost_history)
    plt.plot(cost_history_all, color='yellow')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Function')
    plt.show()


theta = np.zeros((2))
theta, cost_history, train_X, train_y = (
        linear_regression_LASSO(pd.DataFrame(df_norm.sqft_living),
                          df_norm.iloc[:,0], theta, 1, .01, 10000))
predict_Y_for_X1 = train_X.dot(theta)
predict(train_X[:,0], train_y, predict_Y_for_X1)


theta = np.zeros((len(df_norm.columns)))
theta, cost_history_all, train_X, train_y = (
        linear_regression_LASSO(df_norm.drop('price (grands)', axis=1),
                          df_norm.iloc[:,0], theta, 1, .01, 10000))
predict_Y = train_X.dot(theta)
plt.scatter(predict_Y, train_y, color='blue', linewidth=1)
plt.show()



print("\n\n1.5 For lasso Momentum")
momentum = {0.1, 0.5, 0.7, 0.9}
for momenta in momentum:
    theta = np.zeros((2))
    theta, cost_history, train_X, train_y = (
            linear_regression_LASSO_momentum(pd.DataFrame(df_norm.sqft_living),
                     df_norm.iloc[:,0], theta, 1, .01, momenta, 10000))
    predict_Y_for_X1 = train_X.dot(theta)
    predict(train_X[:,0], train_y, predict_Y_for_X1)
    print("Mean square error for X_1 using lasso {0}".format(
            ((predict_Y_for_X1 - train_y) ** 2).mean()))
    
    theta = np.zeros((len(df_norm.columns)))
    theta, cost_history_all, train_X, train_y = (
            linear_regression_LASSO_momentum(df_norm.drop('price (grands)', axis=1),
                     df_norm.iloc[:,0], theta, 1, .01, momenta, 10000))
    predict_Y = train_X.dot(theta)
    print("Mean square error for X using lasso {0}".format(
            ((predict_Y - train_y) ** 2).mean()))
    
    plt.yscale('log')
    plt.plot(cost_history)
    plt.plot(cost_history_all, color='yellow')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Function')
    plt.show()