# Building a Convolutional Neural Network to classifiy computer 
# attacks as types of malware
# imports
import keras
import pandas as pd
import torch_cnn as tcnn
import keras_cnn as kcnn
import data_processing as d
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 


### Data Retieval
file='MachineLearningProject/datasets/VirusSample.csv'
X, y = d.get_encoded_data(file)

# Keras CNN Data
X_train, X_test, y_train, y_test = d.splitData(X, y)

# Convert data to Tensors to Incease Computation time for Torch CNN
X_tensor_train,Y_tensor_train  = d.toTensors(X, y)

# Change Y data for SVM
svm_X_train, svm_X_test, svm_y_train, svm_y_test =  train_test_split(X, d.get_SVM_y(file), test_size=0.33, random_state=1)



### Variables
EPOCHS = 1000 # arbitrary

# Keras CNN
LAYERS = 52 # arbitrary number for now

# Torch CNN
N_FEATURES = X_train.shape[1] # Number of features for the input layer
NUM_ROWS_TRAINING = X_train.shape[0] # Number of rows
N_HIDDEN = N_FEATURES * 10 # Size of first linear layer
N_CNN_KERNEL, MAX_POOL_KERNEL = 3, 4 # CNN kernel size



### Torch CNN
# Build CNN
net = tcnn.CNN(n_feature=N_FEATURES, n_hidden=N_HIDDEN, n_output=13, n_cnn_kernel=N_CNN_KERNEL)   # define the network    
# Train and Test
train_results, results = tcnn.trainTestCNN(net,X_tensor_train,Y_tensor_train)
# Plotting
for metric in train_results:
    kcnn.plot(train_results, EPOCHS, metric, "Training")
for metric in results:
    kcnn.plot(results, EPOCHS/100, metric, "Testing")



### SVM
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
accuracy = []
# Train and Test
for x in range(0,EPOCHS):
    clf.fit(svm_X_train,  svm_y_train)
    if x % 100 == 0:
        predictions=clf.predict(svm_X_test)
        accuracy.append(accuracy_score(svm_y_test, predictions))
# Plotting        
results = {'Accuracy':accuracy}
for metric in results:
    kcnn.plot(results, EPOCHS/100, metric, "Testing")        