"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2016

            **  Tweets Classifier  **

Auther(s): Tomer Patel, Dan Kufra, Gilad Wolf

===================================================
"""
from load_tweets import load_dataset
import operator
from sklearn import svm
import numpy as np


LOW_APPEARANCE = 3

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

    def general_train(self, instances, labels):
        pass

    def word_count(self, tweets, name):
        """
        Recieves a list of tweets for a person
        :param tweets: A list of strings, that represents a tweet
        :param name: The person who wrote all of those tweets
        :return: dictionary, vector - A dictionary of words that appear in a
        tweet and how many times it appear in all the tweets
        """
        my_dict = {}
        # count all the words in all the tweets
        for tweet in tweets:
            tweet = tweet.lower().split()
            for word in tweet:
                if word in my_dict:
                    my_dict[word] += 1
                else:
                    my_dict[word] = 1

        # remove the words that appears the less
        my_dict = {k:v for k,v in my_dict.items() if v > LOW_APPEARANCE}

        index = 0
        for k in my_dict:
            my_dict[k] = (my_dict[k], index)
            index += 1
        return my_dict

    def get_tweet_vec(self, tweet, word_dic):
        # given a tweet and a word_dic it creates a vector of 0's and 1's
        tweet_vec = np.zeros(len(word_dic))
        for word in tweet:
            temp = word_dic.get(word)
            if temp is not None:
                index = temp[1]
                tweet_vec[index] = 1
        return tweet_vec

    def create_tweet_matrix(self, instances, labels):
        pass


    def train_politics(self,instances, labels):
        pass

    def predict_politics(self, tweet, SVC):
        pass




X,y = load_dataset("/cs/hackathon/dan_kufra/venv/IML_Hacakthon/tweets.csv")
training_instances = X[0:0.7*len(X)]
validation_instances = X[0.7*len(X):]
training_labels = y[0:0.7*len(y)]
validation_labels = y[0.7*len(y):]
SVC = svm.SVC()
X,y = load_dataset()
names = np.loadtxt("names.txt", delimiter=',')
print(X[0], names[y[0]])
