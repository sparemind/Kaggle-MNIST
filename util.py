# Contains utilities for loading data, dividing it into training and validation
# sets, and evaluating the accuracy of a model.


from time import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump, load


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
# features and labels.
def split_data(raw_data):
    N = raw_data.shape[0]
    # Randomize and create final validation and training sets
    index_permutation = np.random.permutation(N)
    raw_data = raw_data[index_permutation,:]
    val_index = (int) (N * VAL_SIZE)

    labels_val = raw_data[:val_index,0]
    x_val = raw_data[:val_index,1:]
    labels_train = raw_data[val_index:,0]
    x_train = raw_data[val_index:,1:]

    return x_train, labels_train, x_val, labels_val

