# Building a Convolutional Neural Network to classifiy computer 
# attacks as types of malware
# imports
import torch
import matplotlib.pyplot as plt
import tensor_cnn as tcnn
import data_processing as d  
from sklearn.metrics import log_loss, roc_curve, auc

X, y = d.get_encoded_data()
X_train, X_test, y_train, y_test = d.splitData(X, y)
# Convert data to Tensors to Incease Computation time for CNN
X_tensor_train,Y_tensor_train,X_tensor_test,Y_tensor_test  = d.toTensors(X_train, y_train, X_test, y_test)

# CNN Variables - Dimensions
# Number of features for the input layer
N_FEATURES = X_train.shape[1]
# Number of rows
NUM_ROWS_TRAINING = X_train.shape[0]
# Size of first linear layer
N_HIDDEN = N_FEATURES * 10
# CNN kernel size
N_CNN_KERNEL, MAX_POOL_KERNEL = 3, 4
# Number of Iterations over dataset
EPOCHS=100 #500 
    
# Build CNN
net = tcnn.CNN(n_feature=N_FEATURES, n_hidden=N_HIDDEN, n_output=13, n_cnn_kernel=N_CNN_KERNEL)   # define the network    
# Train and Test
all_losses,pred_y,target_y = tcnn.trainTestCNN(net,X_tensor_train,Y_tensor_train,X_tensor_test,Y_tensor_test)

plt.title('Losses')
plt.plot(all_losses)
plt.show()

false_positive_rate, true_positive_rate, thresholds = roc_curve(target_y,pred_y)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('LOG_LOSS=' + str(log_loss(target_y, pred_y)))
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.6f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()