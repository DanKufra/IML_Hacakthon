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
        vectorizer = TfidfVectorizer(min_df=7)
        X_train = vectorizer.fit_transform(training_instances)
        y = np.array([i for i in training_labels], dtype=float)

        x = tf.placeholder(tf.float64, shape=X_train.A.shape)
        y_ = tf.placeholder(tf.float64, shape=y.shape)

        w = tf.Variable(tf.random_normal([1], stddev=0.35))

        p = w * X_train.A.transpose()
        r = p - y
        s = tf.square(r)

        loss = tf.reduce_max(s)

        cross_entropy = -tf.reduce_sum(y_*tf.log(y))
        train = tf.train.AdagradOptimizer(0.01).minimize(loss)
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())

        for step in range(300):
            sess.run(w)
            train.run(feed_dict={x: X_train.A, y_: y})
            if step % 10:
                print(step, sess.run(w))

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        X_val = vectorizer.fit_transform(validate_instances)
        y_val = np.array([i for i in validate_labels])

        x = tf.placeholder(tf.float64, shape=X_val.A.shape)
        y_ = tf.placeholder(tf.float64, shape=y_val.shape)

        print(accuracy.eval(feed_dict={x: X_val.A, y_: y_val}))

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

training_data = X[0:(int) (0.5*len(X))]
val_data = X[(int) (0.5*len(X)):(int) (1.0*len(X)):]

training_label = y[0:(int) (0.5*len(X))]
val_label = y[(int) (0.5*len(X)):(int) (1.0*len(X)):]

classifier = Classifier()
classifier.train(training_data, training_label, val_data, val_label)

#print (classifier.test_training(val_data, val_label , classifier.first_SVC))

'''
Scale the instances (normalize them to between 0 and 1)

'''