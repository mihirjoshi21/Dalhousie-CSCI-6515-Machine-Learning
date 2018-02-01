# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:56:26 2017

@author: mj
"""

import numpy as np
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.datasets as skd
import sklearn.svm as svm 
import sklearn.utils as sku

iris = skd.load_iris()

svc_iris = svm.SVC(kernel='linear' ,C=1).fit(iris.data, iris.target)
svc_sepal = svm.SVC(kernel='linear' ,C=1).fit(iris.data[:, 0:2], iris.target)
svc_petal = svm.SVC(kernel='linear' ,C=1).fit(iris.data[:, 2:4], iris.target)

predicted_iris = svc_iris.predict(iris.data)
predicted_sepal = svc_sepal.predict(iris.data[:, 0:2])
predicted_petal = svc_petal.predict(iris.data[:, 2:4])

# Fit is used to fit the SVM model based on data provided and predict 
# is for classfiying that data.

# 1.2
print("Accuracy of predicted_iris {0}".format(skm.accuracy_score(iris.target, predicted_iris)))
print("Accuracy of predicted_sepal {0}".format(skm.accuracy_score(iris.target, predicted_sepal)))
print("Accuracy of predicted_petal {0}".format(skm.accuracy_score(iris.target, predicted_petal)))

print("Precision score of predicted_iris {0}".format(skm.precision_score(iris.target,
      predicted_iris, average='weighted')))
print("Precision score of predicted_sepal {0}".format(skm.precision_score(iris.target,
      predicted_sepal, average='weighted')))
print("Precision score of predicted_petal {0}".format(skm.precision_score(iris.target,
      predicted_petal, average='weighted')))

print("Recall score of predicted_iris {0}".format(skm.recall_score(iris.target,
      predicted_iris, average='weighted')))
print("Recall score  of predicted_sepal {0}".format(skm.recall_score(iris.target,
      predicted_sepal, average='weighted')))
print("Recall score  of predicted_petal {0}".format(skm.recall_score(iris.target,
      predicted_petal, average='weighted')))

print("F1 score of predicted_iris {0}".format(skm.f1_score(iris.target,
      predicted_iris, average='weighted')))
print("F1 score of predicted_sepal {0}".format(skm.f1_score(iris.target,
      predicted_sepal, average='weighted')))
print("F1 score of predicted_petal {0}".format(skm.f1_score(iris.target,
      predicted_petal, average='weighted')))

print("Confusion matrix of predicted_iris {0}".format(skm.confusion_matrix(iris.target,
      predicted_iris)))
print("Confusion matrix of predicted_sepal {0}".format(skm.confusion_matrix(iris.target,
      predicted_sepal)))
print("Confusion matrix of predicted_petal {0}".format(skm.confusion_matrix(iris.target,
      predicted_petal)))


#1.3
estimator = svm.SVC(kernel='linear', C=1)
cv_10 = skms.cross_val_score(estimator, iris.data, iris.target, cv=10)
print("10 fold cross validation mean {0}".format(cv_10.mean()))
print("10 fold cross validation standard deviation {0}".format(cv_10.std()))

cv_5 = skms.cross_val_score(estimator, iris.data, iris.target, cv=5)
print("5 fold cross validation mean {0}".format(cv_5.mean()))
print("5 fold cross validation standard deviation {0}".format(cv_5.std()))

# 1.4
k_fold = 5;
fold_data, fold_target = sku.shuffle(iris.data, iris.target, random_state=1)
cv_custom = np.zeros(k_fold)  

size = fold_target.size/k_fold

for i in range(0, k_fold):
    svc_fold = None
    for j in range(0, k_fold):
        if i != j:
            if svc_fold is None:
                svc_fold = estimator.fit(fold_data[int(j*size):int(j*size+size), :],
                                         fold_target[int(j*size):int(j*size+size)])
            else:
                svc_fold.fit(fold_data[int(j*size):int(j*size+size), :],
                                         fold_target[int(j*size):int(j*size+size)])
    
    cv_custom[i] = skm.accuracy_score(fold_target[int(i*size):int(i*size+size)],
             svc_fold.predict(fold_data[int(i*size):int(i*size+size), :]))


print("Custom fold mean {0}".format(cv_custom.mean()))
print("Custom fold std {0}".format(cv_custom.std()))