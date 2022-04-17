# imports          
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer                    
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

# Read in csv file
def read_csv():
    df = pd.read_csv('MachineLearningProject/datasets/VirusSample.csv')
    return df


def get_data(): 
    # Read in csv file
    df = read_csv()
    return df["api"].values, df['class'].values

def get_encoded_data():
    # Variables
    y_data = np.zeros((9795, 13)) # (9795,13)
    list_of_classes =['Adware', 'Agent', 'Backdoor', 'Trojan', 'Virus', 'Worms', 'Downloader', 'Spyware', 'Ransomware', 'Riskware', 'Dropper', 'Crypt', 'Keylogger']
    x,y = get_data()

    # tokenizing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    x_train = tokenizer.texts_to_sequences(x) 
    # Pad sequences with zeros
    x_train = pad_sequences(x_train, padding='post', maxlen=100)

    count = 0
    for z in y:
        y_data[count][list_of_classes.index(z)] = 1
        count+=1

    return x_train,y_data


def encodeData(X , y):
  y = LabelEncoder().fit_transform(y)
  onehot_encoded = OneHotEncoder(handle_unknown="ignore",sparse=False).fit_transform(X)
  print(y.shape())
  return onehot_encoded, y

