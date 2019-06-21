# Contains utilities for loading data, dividing it into training and validation
# sets, and evaluating the accuracy of a model.


from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Files containing training and testing data
TRAIN_DATA = 'data/train.csv'
TEST_DATA  = 'data/test.csv'
# Fraction of the training data that will be used for validation
VAL_SIZE = 0.2

# Accepts the filepath to a .csv file whose contents are then loaded and
# returned as a NumPy array. The filename and load time are also logged to 
# the console.
def load_csv(filepath):
    start = time()
    print(f'Loading {filepath}...', end='', flush=True)
    data = pd.io.parsers.read_csv(filepath).values
    load_time = time() - start
    print(f'DONE ({load_time:.2f}s)')
    return data

# Returns the percentage of correct predictions for some expected values.
def accuracy(expected, predicted):
    return np.mean(expected == predicted)

# Splits the data into a training set and validation set and returns both as
# features and labels. Can provide an optional seed that will be used when
# randomly permuting the data.
def split_data(raw_data, seed=None):
    x = raw_data[:,1:]
    y = raw_data[:,0]
    return train_test_split(x, y, test_size=VAL_SIZE, random_state=1234)

