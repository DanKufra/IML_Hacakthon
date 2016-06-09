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
from sklearn import svm
import numpy as np
from collections import Counter


LOW_APPEARANCE = 3
POLITICIAN = 3


class Classifier(object):
    first_dic = {}
    first_SVC = None
    def __init__(self):
        pass

    def classify(self,X):
        """
        Recieves a list of m unclassified tweets, and predicts for each one which celebrity posted it.
        :param X: A list of length m containing the tweet's texts (strings)
        :return: y_hat - a vector of length m that contains integers between 0 - 9
        """
        # TODO implement

    def general_train(self, training_instances, training_labels):
        self.first_dic = self.get_dict(training_instances)
        X1,y1 = self.create_tweet_matrix(training_instances, training_labels)

    def get_dict(self, tweets):
        """
        Recieves a list of tweets for a person
        :param tweets: A list of strings, that represents a tweet
        :param name: The person who wrote all of those tweets
        :return: dictionary, vector - A dictionary of words that appear in a
        tweet and how many times it appear in all the tweets
        """
        all_tweets = ""
        # count all the words in all the tweets
        for tweet in tweets:
            tweet = tweet.lower()
            all_tweets += " " + tweet
        counter = Counter(all_tweets.split()).most_common()
        my_dict = dict(counter)

        # remove the words that appears the less
        my_dict = {k:v for k,v in my_dict.items() if v > LOW_APPEARANCE}

        #get a list of all the Prepostion in the english language
        get_Prepostion_List = [line.rstrip('\n') for line in open('PrepositionsList')]

        #deletes from the dictionary all prepostions
        my_dict = {k:v for k,v in my_dict.items() if k not in get_Prepostion_List}

        print(my_dict)

        index = 0
        for k in my_dict:
            my_dict[k] = (my_dict[k], index)
            index += 1
        return my_dict

    def get_tweet_vec(self, tweet, word_dic):
        # given a tweet and a word_dic it creates a vector with the number of
        # appearance of a word in the tweet
        tweet_vec = np.zeros(len(word_dic))
        for word in tweet:
            temp = word_dic.get(word)
            if temp is not None:
                index = temp[1]
                tweet_vec[index] += 1
        return tweet_vec

    def create_tweet_matrix(self, instances, labels):
        """
        create matrix of tweets and a vector of their labels
        :param instances: all the tweets in a list of vectors
        :param labels: the labels of the tweeters
        :return:topple of a matrix for all the tweets and a label 1 if politician, else -1
        """
        matrix = np.array([self.get_tweet_vec(tweet, self.first_dic) for tweet in instances])
        binary_labels = np.array([1 if label <= POLITICIAN else -1 for label in
                                 labels])
        return matrix, binary_labels

    def train_politics(self,instances, labels):
        X,y = self.create_tweet_matrix(instances,labels)
        SVC = svm.SVC()
        SVC.fit(X, y)

    def predict_politics(self, tweet, SVC):
        return SVC.predict(self.get_tweet_vec(tweet, self.first_dic))

    def test_training(self, test_instances, test_labels, SVC):
        X2, y2 = self.create_tweet_matrix(test_instances, test_labels)
        return ((1.0 for i in range(len(test_instances)) if self.predict_politics(X2[i], SVC) != y2[i]) / len(test_instances))


X,y = load_dataset()
names = pandas.read_csv("names.txt", header=None)
namesIndex, names = names[0], names[1]
classifier = Classifier()
classifier.general_train(X,y)
