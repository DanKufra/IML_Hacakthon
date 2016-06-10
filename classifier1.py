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
from check import *
from sklearn.multiclass import OneVsRestClassifier
from nltk import word_tokenize

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

    def train(self, training_instances, training_labels,test_instances, test_labels):
        self.vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,6), analyzer='char_wb', lowercase=False, strip_accents="unicode")
        X = self.vectorizer.fit_transform(training_instances)
        classifier = OneVsRestClassifier(svm.SVC(kernel='sigmoid'), 3)
        classifier.fit(X, training_labels)
        X2 = self.vectorizer.transform(test_instances)
        print (classifier.score(X2,test_labels))


X,y = load_dataset()
training_data = X[0:(int) (0.8*len(X))]
val_data = X[(int) (0.8*len(X)):(int) (1.0*len(X)):]

training_label = y[0:(int) (0.8*len(X))]
val_label = y[(int) (0.8*len(X)):(int) (1.0*len(X)):]

classifier = Classifier()
classifier.train(training_data, training_label, val_data, val_label)

'''
Scale the instances (normalize them to between 0 and 1)

'''