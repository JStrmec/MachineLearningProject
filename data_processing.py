# imports 
import torch         
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer   
from sklearn.model_selection import train_test_split                  
from keras.preprocessing.sequence import pad_sequences

# Read in csv file
def read_csv(file):
    df = pd.read_csv(file)#'MachineLearningProject/datasets/VirusSample.csv'
    return df


def get_SVM_y(file ='datasets/VirusSample.csv'):
    df = read_csv(file)
    y = LabelEncoder().fit_transform(df['class'].values)
    return  y


def get_data(file): 
    # Read in csv file
    df = shuffle(read_csv(file))
    df = shuffle(df)
    return df["api"].values, df['class'].values


def get_encoded_data(file ='datasets/VirusSample.csv'):
    # Variables
    #len(set_api)=(7966)
    y_data = np.zeros((9795, 13)) # (9795,13)
    list_of_classes =['Adware', 'Agent', 'Backdoor', 'Trojan', 'Virus', 'Worms', 'Downloader', 'Spyware', 'Ransomware', 'Riskware', 'Dropper', 'Crypt', 'Keylogger']
    x ,y = get_data(file)

    # tokenizing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    x_train = tokenizer.texts_to_sequences(x) 
    # Pad sequences with zeros
    x_train = pad_sequences(x_train, padding='post', maxlen=50)

    count = 0
    for z in y:
        y_data[count][list_of_classes.index(z)] = 1
        count+=1

    return x_train,y_data

def standarize_predictions(pred):
    for x in range(0,len(pred)):
        z = pred[x]
        max_val = max(z)
        for y in range(0,len(z)):
            if pred[x][y] != max_val:
                pred[x][y] = 0
            else:
                pred[x][y] = 1
    return pred

def XnumpyToTensor(x_data_np):
    x_tensor = Variable(torch.from_numpy(x_data_np).type(torch.FloatTensor)) # Note the conversion for pytorch
    return x_tensor

def YnumpyToTensor(y_data_np):    
    y_tensor = Variable(torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float        
    return y_tensor


def splitData(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

def toTensors(x, y):
    x_tensor_train = XnumpyToTensor(x)
    y_tensor_train = YnumpyToTensor(y)
    # x_tensor_test = XnumpyToTensor(x_test)
    # y_tensor_test = YnumpyToTensor(y_test)
    return x_tensor_train,y_tensor_train#,x_tensor_test,y_tensor_test 
