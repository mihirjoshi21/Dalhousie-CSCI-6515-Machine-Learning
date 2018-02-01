# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 03:14:41 2017

@author: mj
"""

import sklearn.metrics as skm
import sklearn.datasets as skd
import sklearn.feature_extraction.text as skfet
import sklearn.ensemble as ske
import sklearn.pipeline as skp
import sklearn.neural_network as skn

#2.1
news = skd.fetch_20newsgroups(subset = 'all', shuffle = True, 
                                               random_state = 90)
train_20news = skd.fetch_20newsgroups(subset = 'train', shuffle = True, 
                                               random_state = 90)

test_20news = skd.fetch_20newsgroups(subset='test', shuffle = False,
                                          random_state = 100)

                                     
# 2.2
vectorizer = skfet.CountVectorizer()
train_vectors = vectorizer.fit_transform(train_20news.data, train_20news.target)
test_vectors = vectorizer.transform(test_20news.data)

vectorizer = skfet.TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_20news.data, train_20news.target)
test_vectors = vectorizer.transform(test_20news.data)

# 2.3
RF = ske.RandomForestClassifier(n_estimators=50)
predict = RF.fit(train_vectors, train_20news.target).predict(test_vectors)

print("Accuracy of Random Forest {0}".format(
        skm.accuracy_score(test_20news.target, predict)))
print("Precision score of Random Forest {0}".format(
        skm.precision_score(test_20news.target, predict, average='weighted')))
print("Recall score of Random Forest {0}".format(
        skm.recall_score(test_20news.target, predict, average='weighted')))
print("f1 score score of Random Forest {0}".format(
        skm.f1_score(test_20news.target, predict, average='weighted')))
print("Confusion matrix of Random Forest {0}".format(
        skm.confusion_matrix(test_20news.target, predict)))

# 2.4
pipeline = skp.Pipeline([
    ('vect', skfet.CountVectorizer()),
    ('tfidf', skfet.TfidfTransformer())
])
train_vectors = pipeline.fit_transform(train_20news.data, train_20news.target)
test_vectors = pipeline.transform(test_20news.data)
predict_pipeline = RF.fit(train_vectors,
                          train_20news.target).predict(test_vectors)

print("Accuracy after pipleine {0}".format(
        skm.accuracy_score(test_20news.target, predict_pipeline)))
print("Precision after pipleine {0}".format(
        skm.precision_score(test_20news.target, predict_pipeline, average='weighted')))
print("Recall score after pipleine {0}".format(
        skm.recall_score(test_20news.target, predict_pipeline, average='weighted')))
print("f1 score after pipleine {0}".format(
        skm.f1_score(test_20news.target, predict_pipeline, average='weighted')))
print("Confusion matrix after pipleine{0}".format(
        skm.confusion_matrix(test_20news.target, predict_pipeline)))

#2.5
mlp = skn.MLPClassifier(hidden_layer_sizes=(10, 20, 10), max_iter=10, random_state=1)
predict_nn = mlp.fit(train_vectors, train_20news.target).predict(test_vectors)
print("Accuracy after MLPClassifier {0}".format(
        skm.accuracy_score(test_20news.target, predict_nn)))
print("Precision after MLPClassifier {0}".format(
        skm.precision_score(test_20news.target, predict_nn, average='weighted')))
print("Recall score after MLPClassifier {0}".format(
        skm.recall_score(test_20news.target, predict_nn, average='weighted')))
print("f1 score after MLPClassifier {0}".format(
        skm.f1_score(test_20news.target, predict_nn, average='weighted')))
print("Confusion matrix after MLPClassifier {0}".format(
        skm.confusion_matrix(test_20news.target, predict_nn)))