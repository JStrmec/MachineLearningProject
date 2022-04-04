# Building a Convolutional Neural Network to classifiy computer 
# attacks as types of malware
import torch.nn as nn
import torch
import data_processing as d
from sklearn.model_selection import train_test_split


X, y = d.get_encoded_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# References:
# https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/5_convolutional_net.py
# https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556

# use_cuda=False
X_tensor_train= XnumpyToTensor(trainX) # default order is NBC for a 3d tensor, but we have a 2d tensor
X_shape=X_tensor_train.data.size()

# Dimensions
# Number of features for the input layer
N_FEATURES=trainX.shape[1]
# Number of rows
NUM_ROWS_TRAINNING=trainX.shape[0]
# this number has no meaning except for being divisable by 2
N_MULT_FACTOR=8 # min should be 4
# Size of first linear layer
N_HIDDEN=N_FEATURES * N_MULT_FACTOR
# CNN kernel size
N_CNN_KERNEL=3
MAX_POOL_KERNEL=4

DEBUG_ON=False

def debug(x):
    if DEBUG_ON:
        print ('(x.size():' + str (x.size()))
    
class Net2(nn.Module):    
    def __init__(self, n_feature, n_hidden, n_output, n_cnn_kernel, n_mult_factor=N_MULT_FACTOR):
        super(Net2, self).__init__()
        self.n_feature=n_feature
        self.n_hidden=n_hidden
        self.n_output= n_output 
        self.n_cnn_kernel=n_cnn_kernel
        self.n_mult_factor=n_mult_factor
        self.n_l2_hidden=self.n_hidden * (self.n_mult_factor - self.n_cnn_kernel + 3)
#         self.n_out_hidden=int (self.n_l2_hidden/2)
                        
        self.l1 = nn.Sequential(
            torch.nn.Linear(self.n_feature, self.n_hidden),
            torch.nn.Dropout(p=1 -.85),            
            torch.nn.LeakyReLU (0.1),            
            torch.nn.BatchNorm1d(self.n_hidden, eps=1e-05, momentum=0.1, affine=True)            
        )                
        self.c1= nn.Sequential(            
            torch.nn.Conv1d(self.n_feature, self.n_hidden, 
                            kernel_size=(self.n_cnn_kernel,), stride=(1,), padding=(1,)),
            torch.nn.Dropout(p=1 -.75),            
            torch.nn.LeakyReLU (0.1),
            torch.nn.BatchNorm1d(self.n_hidden, eps=1e-05, momentum=0.1, affine=True)        
        )                        
        self.out = nn.Sequential(
            torch.nn.Linear(self.n_l2_hidden,
                            self.n_output),  
        )                
        self.sig=nn.Sigmoid()

        
    def forward(self, x):
        debug(x)
        varSize=x.data.shape[0] # must be calculated here in forward() since its is a dynamic size        
        x=self.l1(x)                
        debug(x)
        # for CNN        
        x = x.view(varSize,self.n_feature,self.n_mult_factor)
        debug(x)
        x=self.c1(x)
        debug(x)
        # for Linear layer
        x = x.view(varSize, self.n_hidden * (self.n_mult_factor -self.n_cnn_kernel + 3))
        debug(x)
#         x=self.l2(x)                    
        x=self.out(x)   
        debug(x)
        x=self.sig(x)
        return x
    
net = Net2(n_feature=N_FEATURES, n_hidden=N_HIDDEN, n_output=1, n_cnn_kernel=N_CNN_KERNEL)   # define the network    
if use_cuda:
    net=net.cuda() # very important !!!
lgr.info(net)
b = net(X_tensor_train)
print ('(b.size():' + str (b.size())) # torch.Size([108405, 928])