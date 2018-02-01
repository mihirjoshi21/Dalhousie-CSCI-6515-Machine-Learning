# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 18:14:35 2017

@author: mj
"""

import numpy as np
import sklearn.model_selection as skms
import sklearn.svm as svm 


wine_data = np.loadtxt('wine.train', delimiter=',')

target = wine_data.T[0]    
wine_data = np.delete(wine_data, np.s_[:1:1], 1) 

features_data = np.zeros([120, 11])

features_data[:, 0:2] = wine_data[:, 0:2]
features_data[:, 2:11] = wine_data[:, 3:12]

X_train, X_test, y_train, y_test = skms.train_test_split(
        wine_data, target, test_size=0.22, random_state=42)


estimator = svm.SVC(kernel='poly', C=1)

fit_wine = estimator.fit(X_train, y_train)

wine_test_data = np.loadtxt('wine.test', delimiter=',')
wine_test_data = np.delete(wine_test_data, np.s_[:1:1], 1) 
features_test_data = np.zeros([58, 11])
features_test_data[:, 0:2] = wine_test_data[:, 0:2]
features_test_data[:, 2:11] = wine_test_data[:, 3:12]

predict_wine = fit_wine.predict(features_test_data)

np.savetxt('wine_final_classification.csv', predict_wine, delimiter=',', fmt='%d')
