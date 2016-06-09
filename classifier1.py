__author__ = 'dan_kufra'
"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2016

            **  Tweets Classifier  **

Auther(s): Tomer Patel, Dan Kufra, Gilad Wolf

===================================================
"""
import pandas
from load_tweets import load_dataset
import operator
from sklearn import *
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


LOW_APPEARANCE = 5
POLITICIAN = 3


class Classifier(object):

    def __init__(self):
        pass

    def classify(self,X):
        """
        Recieves a list of m unclassified tweets, and predicts for each one which celebrity posted it.
        :param X: A list of length m containing the tweet's texts (strings)
        :return: y_hat - a vector of length m that contains integers between 0 - 9
        """
        # TODO implement

    def train(self,training_instances, training_labels,validate_instances, validate_labels):
        SVC = svm.SVC()
        vectorizer = TfidfVectorizer(min_df=3, stop_words='english')
        X_train = vectorizer.fit_transform(training_instances)
        X_train2 = vectorizer.transform(validate_instances)

        '''
        binary_labels = np.array([1 if label <= POLITICIAN else -1 for label in
                                 training_labels])
        sparse_labels1 = csr_matrix((binary_labels)).toarray()[0]
        SVC.fit(X_train.A, sparse_labels1)
        binary_labels2 = np.array([1 if label <= POLITICIAN else -1 for label in
                                 validate_labels])
        sparse_labels2 = csr_matrix((binary_labels2)).toarray()[0]
        '''

        SVC.fit(X_train.A, training_labels)
        res = SVC.score(X_train2, val_label)
        print(res)
        return res
        



X,y = load_dataset()
names = pandas.read_csv("names.txt", header=None)
namesIndex, names = names[0], names[1]

training_data = X[0:(int) (0.2*len(X))]
val_data = X[(int) (0.2*len(X)):(int) (0.4*len(X)):]

training_label = y[0:(int) (0.2*len(X))]
val_label = y[(int) (0.2*len(X)):(int) (0.4*len(X)):]

classifier = Classifier()
classifier.train(training_data, training_label, val_data, val_label)

#print (classifier.test_training(val_data, val_label , classifier.first_SVC))

'''
Scale the instances (normalize them to between 0 and 1)

'''