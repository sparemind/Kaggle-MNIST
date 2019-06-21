# Multinomial logistic classifier


from util import *
from sklearn.linear_model import LogisticRegression
from joblib import dump, load


# Load data and get training and validation sets
raw_train = load_csv(TRAIN_DATA)
x_train, x_val, y_train, y_val = split_data(raw_train, 1234)
x_train = x_train / 255.0
x_val = x_val / 255.0
print(f'Training set size:   {x_train.shape[0]}')
print(f'Validation set size: {x_val.shape[0]}')

# Train model
start = time()
#clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(x_train, y_train)
#dump(clf, 'logistic.joblib')
clf = load('logistic.joblib')
print(f'Took {time() - start:.2f}s to create model.')

predict_train = clf.predict(x_train)
predict_val = clf.predict(x_val)
accuracy_train = accuracy(y_train, predict_train)
accuracy_val = accuracy(y_val, predict_val)
print(f'Training Accuracy:   {accuracy_train:.5f}')
print(f'Validation Accuracy: {accuracy_val:.5f}')

 

