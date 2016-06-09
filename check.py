__author__ = 'gilax'

from load_tweets import *
from collections import Counter
import numpy as np

X,y = load_dataset()
all_tweets_list = ["" for _ in y]
all_tweets = ""
# count all the words in all the tweets
for i in range(len(X)):
    tweet = X[i].lower()
    all_tweets_list[y[i]] += " " + tweet
    all_tweets += " " + tweet
my_dict = []

for label in range(10):
    counter = Counter(all_tweets_list[label].split()).most_common()
    my_dict.append(counter)

#get a list of all the Prepostion in the english language
get_Prepostion_List = [line.rstrip('\n') for line in open('PrepositionsList')]

my_dict = [{k:v for k,v in my_dict[i] } for i in range(len(my_dict))]

check = open('check.txt', 'w')

names = pandas.read_csv("names.txt", header=None)
namesIndex, names = names[0], names[1]

# count all the words in all the tweets

counter = Counter(all_tweets.split()).most_common()
all_counter = []

# print(counter)

for word in counter:
    to_write = []
    for i in range(10):
        if word[0] in my_dict[i].keys():
            percent = round(my_dict[i][word[0]] / word[1] * 100, 2)
            to_write.append(percent)
        else:
            to_write.append(0)
    check.write(word[0] + to_write.__str__() + "\n")
    # check.write(names[i] + "\n")
    # for string in my_dict[i]:
    #     check.write(string.__str__() + "\n")
    # check.write("\n\n")
