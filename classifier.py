"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2016

            **  Tweets Classifier  **

Author(s): Tomer Patel, Dan Kufra, Gilad Wolf

===================================================
"""
from sklearn import *
from sklearn.feature_extraction.text import TfidfVectorizer
from check import *



class Classifier(object):

    classifier = None

    def __init__(self):
        X,y = load_dataset()
        self.classifier = self.train(X,y)

    def classify(self,X):

        """
        Recieves a list of m unclassified tweets, and predicts for each one which celebrity posted it.
        :param X: A list of length m containing the tweet's texts (strings)
        :return: y_hat - a vector of length m that contains integers between 0 - 9
        """
        X_train = self.vectorizer.transform(X)
        return self.classifier.predict(X_train)

    def train(self, training_instances, training_labels):
        self.vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,6), analyzer='char_wb', lowercase=False, strip_accents='unicode')
        X = self.vectorizer.fit_transform(training_instances)
        classifier = svm.LinearSVC()
        classifier.fit(X, training_labels)
        return classifier




classifier = Classifier()

