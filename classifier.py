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
import numpy as np

class Classifier(object):

    def __init__(self):
        pass

    def classify(self,X):
        """
        Recieves a list of m unclassified tweets, and predicts for each one which celebrity posted it.
        :param X: A list of length m containing the tweet's texts (strings)
        :return: y_hat - a vector of length m that contains integers between 0 - 9
        """

    raise NotImplementedError("TODO: Implement this method by 12pm tomorrow!")

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
        min_val = min(my_dict.values())
        remaining = my_dict.keys() - (k for k, v in my_dict.iteritems() if v == min_val)
        return my_dict
