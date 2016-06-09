import pandas

def load_dataset(filename='tweets.csv'):
    train = pandas.read_csv(filename, header=None)
    X, y = train[1], train[0]
    X = X.tolist()
    return X,y