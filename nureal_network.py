__author__ = 'gilax'

import pandas
from load_tweets import load_dataset
from sklearn import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf


class Classifier(object):
    def classify(self, X):
                """
        Recieves a list of m unclassified tweets, and predicts for each one which celebrity posted it.
        :param X: A list of length m containing the tweet's texts (strings)
        :return: y_hat - a vector of length m that contains integers between 0 - 9
        """
        # TODO implement

    def train(self,training_instances, training_labels, validate_instances, validate_labels):
        vectorizer = TfidfVectorizer(min_df=100)
        X_train = vectorizer.fit_transform(training_instances)
        y = np.array([i for i in training_labels])

        w = tf.Variable(tf.random_normal([len(X_train.A)], stddev=0.35))

        p = X_train.A * w
        r = p - y
        s = tf.square(r)

        loss = tf.reduce_mean(s)
        train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        for step in range(300):
            sess.run(train)
            print(step, sess.run(w))
            if step % 20:
                pass


                # SVC = svm.LinearSVC()
                # vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,4))
                # X_train = vectorizer.fit_transform(training_instances)
                #
                # X_train2 = vectorizer.transform(validate_instances)
                #
                # binary_labels = np.array([1 if label <= POLITICIAN else -1 for label in
                #                          training_labels])
                #
                # binary_labels2 = np.array([1 if label <= POLITICIAN else -1 for label in
                #                          validate_labels])
                # SVC.fit(X_train, binary_labels)
                # res = SVC.score(X_train2, binary_labels2)
                # print(res)
                # return res




X,y = load_dataset()
names = pandas.read_csv("names.txt", header=None)
namesIndex, names = names[0], names[1]

training_data = X[0:(int) (0.4*len(X))]
val_data = X[(int) (0.4*len(X)):(int) (1.0*len(X)):]

training_label = y[0:(int) (0.4*len(X))]
val_label = y[(int) (0.4*len(X)):(int) (1.0*len(X)):]

classifier = Classifier()
classifier.train(training_data, training_label, val_data, val_label)

#print (classifier.test_training(val_data, val_label , classifier.first_SVC))

'''
Scale the instances (normalize them to between 0 and 1)

'''