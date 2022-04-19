# imports          
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer                    
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

# Read in csv file
def read_csv():
    df = pd.read_csv('MachineLearningProject/datasets/VirusSample.csv')
    return df

def get_y():
    df = read_csv()
    y = LabelEncoder().fit_transform(df['class'].values)
    return  y

def get_data(): 
    # Read in csv file
    df = shuffle(read_csv())
    df = shuffle(df)
    return df["api"].values, df['class'].values

def get_encoded_data():
    # Variables
    y_data = np.zeros((9795, 13)) # (9795,13)
    list_of_classes =['Adware', 'Agent', 'Backdoor', 'Trojan', 'Virus', 'Worms', 'Downloader', 'Spyware', 'Ransomware', 'Riskware', 'Dropper', 'Crypt', 'Keylogger']
    x ,y = get_data()

    #df = read_csv()
    #x_train = pd.get_dummies(df, columns = ['api'])

    # tokenizing
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    x_train = tokenizer.texts_to_sequences(x) 
    # Pad sequences with zeros
    x_train = pad_sequences(x_train, padding='post', maxlen=50)

    # # vectorization
    # countvectorizer = CountVectorizer()
    # x_train =countvectorizer.fit_transform(x)

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