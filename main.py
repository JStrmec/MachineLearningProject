# Building a Convolutional Neural Network to classifiy computer 
# attacks as types of malware
# imports
import keras
import pandas as pd
import torch_cnn as tcnn
import keras_cnn as kcnn
import data_processing as d
from tensorflow import keras
import matplotlib.pyplot as plt
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

### Variables
EPOCHS = 10 # arbitrary

# Torch CNN
N_FEATURES = X_train.shape[1] # Number of features for the input layer
NUM_ROWS_TRAINING = X_train.shape[0] # Number of rows
N_HIDDEN = N_FEATURES * 10 # Size of first linear layer
N_CNN_KERNEL, MAX_POOL_KERNEL = 3, 4 # CNN kernel size



import torch_cnn as tcnn
### Torch CNN
# Build CNN
net = tcnn.CNN(n_feature=N_FEATURES, n_hidden=N_HIDDEN, n_output=13, n_cnn_kernel=N_CNN_KERNEL)   # define the network    
# Train and Test
train_results, results = tcnn.trainTestCNN(net,X,y,EPOCHS)
# Plotting
d.plot(results, len(results["Accuracy"]), "Accuracy")
   