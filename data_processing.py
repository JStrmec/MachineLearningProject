# imports
from sklearn.model_selection import train_test_split             
from keras.preprocessing.text import Tokenizer                    
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,LabelBinarizer
import pandas as pd
import numpy as np

# Read in csv file
def read_csv():
    df = pd.read_csv('MachineLearningProject/datasets/VirusSample.csv')
    return df


def get_data(return_x_y = False): 
    # Read in csv file
    df = read_csv()
    if(return_x_y):
        return df["api"].values, df['class'].values
    else:
        df.drop['file']
        return df


def get_encoded_data():
    # Variables
    set_api = set() 
    list_of_classes =['Adware', 'Agent', 'Backdoor', 'Trojan', 'Virus', 'Worms', 'Downloader', 'Spyware', 'Ransomware', 'Riskware', 'Dropper', 'Crypt','Keylogger']
    x_data = []  
    y_data = []
    full_api_values_list =[]
    x_temp = []

    # Read in csv file
    df = read_csv()

    # Extract string column api to split for encoding
    api_col = df["api"]
    class_col = df['class']

    # Loop to split rows of api column 
    for x in api_col:
        temp = x.split(',')
        temp.append('stop')
        x_temp.append(temp)
        full_api_values_list.extend(temp)
        set_api.update(temp)
        
    # Create matrix of 0 to begin encoding with
    x_data = np.zeros((api_col.size, len(set_api))) # (9795,7966)
    y_data = np.zeros((api_col.size, 13)) # (9795,13)

    # Encode x data 
    # Convert set to list to get indexs for next step
    list_api = list(set_api)

    index = 0
    for x in full_api_values_list:
        if(x=='stop'):
            index+=1
        if(index == 9795):
            break
        x_data[index][list_api.index(x)] = 1
    count = 0
    for x in class_col:
        y_data[count][list_of_classes.index(x)] = 1
        count+=1
        
    return x_data, y_data


def tokenize_data():
    df = read_csv()
    x = df['api'].values
    y = df['class'].values

    x_train,x_test,y_train,y_test = train_test_split(
                                                    x, y,  
                                                    test_size=0.25,  
                                                    random_state=1000)

    tokenizer = Tokenizer(num_words=7966)
    tokenizer.fit_on_texts(x_train)

    X_train = tokenizer.texts_to_sequences(x_train)
    X_test = tokenizer.texts_to_sequences(x_test)
    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1                          

    maxlen = 100

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


