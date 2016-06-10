__author__ = 'gilax'

MAX = 0

from load_tweets import *
from collections import Counter
import numpy as np


def get_common_words(X, y, labels, promise):
    """
    returns the common words and the percentage of them in each group
    :param X: tweets
    :param y: names indexes
    :param labels: list of names indexes
    :param promise: percent that promising the separate
    :return: a list of [word, [percent, ..., percent]]
    """
    all_tweets_list = ["" for _ in range(len(labels))]
    all_tweets = ""
    # count all the words in all the tweets
    for i in range(len(X)):
        tweet = X[i].lower()
        all_tweets_list[y[i]] += " " + tweet
        all_tweets += " " + tweet


    list_per_group = []

    for label in range(len(labels)):
        counter = Counter(all_tweets_list[label].split()).most_common()
        list_per_group.append(counter)

    # #get a list of all the Prepostion in the english language
    # get_Prepostion_List = [line.rstrip('\n') for line in open('PrepositionsList')]

    # list of dictionaries for every person with word as key and amount in tweets as value
    dictionary_per_person = [{k: v for k, v in list_per_group[i]}
                                            for i in range(len(list_per_group))]

    check = open('check.txt', 'w')

    # count all the words in all the tweets
    counter = Counter(all_tweets.split()).most_common()
    all_counter = []

    for word, amount in counter:
        to_write = []
        for i in range(len(dictionary_per_person)):
            if word in dictionary_per_person[i].keys():
                percent = round(dictionary_per_person[i][word] / amount * 100, 2)
                to_write.append(percent)
            else:
                to_write.append(0)
        if any(to_write[i] >= promise and dictionary_per_person[i][word] > MAX
               for i in range(len(dictionary_per_person))):
            check.write(word + to_write.__str__() + "\n")
            all_counter.append((word, to_write))

    return all_counter


tweets, names = load_dataset()

politician_labels = [i for i in range(10)]

print(len(get_common_words(tweets, names, politician_labels, 20)))