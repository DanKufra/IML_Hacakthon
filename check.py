__author__ = 'gilax'

from load_tweets import *
from collections import Counter

X,y = load_dataset()
all_tweets = ""
# count all the words in all the tweets
for tweet in X:
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