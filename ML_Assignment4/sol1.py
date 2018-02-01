# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:01:19 2017

@author: mj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Calculate probabilites
def probabilities(data, mu, covariance):
    prob = np.power(np.e, float(-0.5 * (data - mu) * (covariance.I) * ((
            data - mu).T))) / (np.power(2 * np.pi, len(data[0]) / 2) * np.power(np.linalg.det(covariance), 0.5))
    return prob


# Calculate estimation
def E(data, prior, covariance, mu, k, count):
    for j in range(count):
        px = 0
        for i in range(k):
            prob[j, i] = prior[i] * probabilities(data[j, :], mu[i],
                covariance[i])
            px += prob[j, i]
        for i in range(k):
            prob[j, i] /= px
    return prob            

# Calculate Maximization
def M(data, prior, covariance, mu, k, count, features, prob):
    prob_sum = np.mat(sum(prob, axis=0))
    for i in range(k):
        mu[i] = np.mat(np.zeros((1, features)))
        covariance[i] = np.mat(np.zeros((features, features)))
        for j in range(count):
            mu[i] += prob[j, i] * data[j, :]
        mu[i] /= prob_sum[0, i]
        for j in range(count):
            covariance[i] += prob[j, i] * (data[j, :] - mu[i]).T * (
                    data[j, :] - mu[i])
        covariance[i] /= prob_sum[0, i]
        prior[i] = prob_sum[0, i] / count
    return covariance, prior, mu


cluster_count  = 4
iris_data = np.loadtxt('fisheriris.data', delimiter=',')
plotnumbercount=0;
fig = plt.figure(figsize=(9, 6))

for i in range(cluster_count):
    if(i+1<2): continue
    k  = i+1
    df = pd.DataFrame(columns=['sepal length', 'sepal width', 'petal length',
                               'petal width','class', 'Prob'])
    data = np.mat(np.delete(iris_data, np.s_[4::1], 1))
    sample_count, feature_count = np.shape(data)
    avg = [np.average(col) for col in data.T[:-1]]
    target_names = np.genfromtxt('feature.txt', delimiter=',', dtype='str')   
            
    prior = np.ones(k)/k
    mu = [data[i, :] for i in np.random.randint(0, sample_count, size=k)]
    covariance = [np.mat(np.identity(feature_count)) for _ in range(k)]
      
    # calculate prior probailities  
    prob = np.zeros((sample_count, k))
    dif = 1
    threshold = .001
        
    while dif > threshold:
        mu_pre = [item for item in mu]
        prob = E(data, prior, covariance, mu, k, sample_count)
        covariance, prior, mu = M(data, prior, covariance, mu, k,
                                  sample_count, feature_count, prob)
            
        dif = 0
        for i in range(k):
            distance = (mu[i]-mu_pre[i])*(mu[i]-mu_pre[i]).T
            dif += distance[0,0]
            
            
    classification = np.mat(np.zeros((sample_count, 2)))
    for i in range(sample_count):
        classification[i, :] = np.argmax(prob[i, :]), np.amax(prob[i, :])
        temp = [item for item in np.squeeze(
                np.asarray(data[i, :]))] + [np.argmax(prob[i, :]), np.amax(prob[i, :])]
        df.loc[i] = temp
    
    for j in range(k):
        pointsInCluster = data[np.nonzero(classification[:, 0].A == j)[0]]
        df['class'] = pd.to_numeric(df['class'],
                   downcast='signed', errors='coerce')
        
    subplot = fig.add_subplot(231 + plotnumbercount) 
    subplot.scatter(df['sepal length'], df['sepal width'], 24, c=df['class'])
    subplot.set_title('Classes '+ str(k))
    subplot.set_xlabel('sepal length')
    subplot.set_ylabel('sepal width')
    plotnumbercount+=1
    silhouette_avg = silhouette_score(np.mat(df[['sepal length','sepal width', 'petal length', 'petal width']]),
                                      df['class'])
    print("For {0} clusters silhouette average is {1}".format(k, silhouette_avg))

plt.tight_layout()
plt.show()