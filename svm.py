# Support vector machine classifier


from util import *
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
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
#n_est = 10
#clf = BaggingClassifier(SVC(gamma='auto', kernel='linear'), max_samples = 1.0 / n_est, n_estimators=n_est)
clf = SVC(gamma=10, kernel='poly', C=0.001)
#clf.fit(x_train, y_train)
#dump(clf, 'svm.joblib')
clf = load('svm.joblib')
print(f'Took {time() - start:.2f}s to create model.')
print('Support Vectors: ', clf.support_vectors_.shape)

predict_val = clf.predict(x_val)
accuracy_val = accuracy(y_val, predict_val)
print(f'Validation Accuracy: {accuracy_val:.5f}')
predict_train = clf.predict(x_train)
accuracy_train = accuracy(y_train, predict_train)
print(f'Training Accuracy:   {accuracy_train:.5f}')
 

