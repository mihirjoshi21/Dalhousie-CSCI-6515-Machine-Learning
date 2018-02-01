# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 20:08:52 2017

@author: mj
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import operator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score    

class BernoulliNaiveBayes():
    def __init__(self, laplace=1):
        self._laplace = laplace
        self.priors = {}
        self.wordCountAccClass = {}
        self.conditionalP = {}
    
    def trainBernoulliNB(self, X, y):
        print('In Training - May take a minute depending on system')
        self.calculatePriors(y)
        self.wordCountAccordingToClass(X, y)
        self.conditionalProbability(X, y)
           
    def calculatePriors(self, y):
        """Calculate prior probabilities of  classes
        sum of total classes of each type/ total target set
        """ 
        total_items = len(y)
        for i in range(0, total_items):
            value = y[i]
            if(value not in self.priors):
                self.priors[value] = 1/total_items
            else:
                self.priors[value] += 1/total_items
                
    def wordCountAccordingToClass(self, X, y):
        """Calculate total number of words in each class
        """ 
        for i in range(0, X.shape[0]):
            value = y[i]
            if(value not in self.wordCountAccClass):
                self.wordCountAccClass[value] = X[i].sum()
            else:
                self.wordCountAccClass[value] += X[i].sum()
                
    def conditionalProbability(self, X, y):
        """ Conditional probability 
            P(X|class) = number_of_X_in_class + laplace / N_words_in_class + N words
        """
        for i in range(0, len(y)):
            classs = y[i]
            if(classs not in self.conditionalP):
                self.conditionalP[classs] = {}   
            self.countEachItem(classs, i, X[i].T)
                
    def countEachItem(self, classs, i, X):
        for i in range(0, len(X)):
            word = X[i]
            if(i not in self.conditionalP[classs]):
                self.conditionalP[classs][i] = word
            else:
                self.conditionalP[classs][i] += word
         
    def testBernoulliNB(self, X):
        print('In Testing')
        predict = np.empty(X.shape[0])
        total_words = X.shape[1]
        for i in range(0, X.shape[0]):
            x = X[i].T
            temp = {}
            for key, value in self.priors.items():
                temp[key] = value
                for j in range(0, len(x)):
                    if(x[j] == 0):
                        continue
                    temp[key] *= (self.conditionalP[key][j] +
                        self._laplace)/(self.wordCountAccClass[key] + total_words)
            predict[i] = max(temp.items(), key=operator.itemgetter(1))[0]
        return predict
            
            


categories = ['alt.atheism', 'soc.religion.christian', 
              'comp.graphics', 'sci.med']
remove = ('headers', 'footers', 'quotes')

train_20news = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=90,
                                remove=remove)

test_20news = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=100,
                               remove=remove)
y_train, y_test = train_20news.target, test_20news.target

vectorizer = CountVectorizer(stop_words='english', binary=True)
X_train = vectorizer.fit_transform(train_20news.data).toarray()
X_test = vectorizer.transform(test_20news.data).toarray()
feature_names = vectorizer.get_feature_names()

alpha = 1
clf = BernoulliNaiveBayes(alpha)
clf.trainBernoulliNB(X_train,y_train)
y_pred = clf.testBernoulliNB(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names = test_20news.target_names))
print(confusion_matrix(y_test, y_pred))