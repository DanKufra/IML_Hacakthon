__author__ = 'gilax'

MAX = 4
PROMISS_PERCENT = 95

from load_tweets import *
from collections import Counter
import numpy as np

X,y = load_dataset()

labels = [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]]

all_tweets_list = ["" for _ in range(len(labels))]
all_tweets = ""
# count all the words in all the tweets
for i in range(len(X)):
    tweet = X[i].lower()
    index = y[i]
    if index in labels[0]:
        all_tweets_list[0] += " " + tweet
    else:
        all_tweets_list[1] += " " + tweet
    all_tweets += " " + tweet

list_per_group = []

for label in range(len(labels)):
    counter = Counter(all_tweets_list[label].split()).most_common()
    list_per_group.append(counter)

# #get a list of all the Prepostion in the english language
# get_Prepostion_List = [line.rstrip('\n') for line in open('PrepositionsList')]

# list of dictionaries for every person with word as key and amount in tweets as value
dictionary_per_group = [{k: v for k, v in list_per_group[i]}
                                        for i in range(len(list_per_group))]

check = open('check.txt', 'w')

names = pandas.read_csv("names.txt", header=None)
namesIndex, names = names[0], names[1]

# count all the words in all the tweets
counter = Counter(all_tweets.split()).most_common()
all_counter = []

# print(counter)
count0 = 0
count1 = 0
for word, amount in counter:
    to_write = [0,0]

    for i in range(len(dictionary_per_group)):
        if word in dictionary_per_group[i].keys():
            percent = round(dictionary_per_group[i][word] / amount * 100, 2)
            percent = dictionary_per_group[i][word]
            to_write.append(percent)
            # if i in labels[0]:
            #     to_write[0] += percent
            # else:
            #     to_write[1] += percent

       # else:
            #to_write.append(0)
    to_write[0] = round(to_write[0])
    to_write[1] = round(to_write[1])

    if to_write[0] >= PROMISS_PERCENT or to_write[1] >= PROMISS_PERCENT:
        if to_write[0] >= PROMISS_PERCENT:
            count0 += 1
        else:
            count1+=1
        check.write(word + to_write.__str__() + "\n")
        all_counter.append((word, to_write))
    # check.write(names[i] + "\n")
    # for string in my_dict[i]:
    #     check.write(string.__str__() + "\n")
    # check.write("\n\n")