# imports
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler,LabelBinarizer
import pandas as pd
import numpy as np

def get_encoded_data():
    # Variables
    set_api = set() 
    list_of_classes =['Adware', 'Agent', 'Backdoor', 'Trojan', 'Virus', 'Worms', 'Downloader', 'Spyware', 'Ransomware', 'Riskware', 'Dropper', 'Crypt','Keylogger']
    x_data = []  
    y_data = []
    full_api_values_list =[]
    x_temp = []

    # Read in csv file
    df = pd.read_csv('MachineLearningProject/datasets/VirusSample.csv')

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
    y_data = np.zeros((api_col.size, 13)) # (9795,7966)

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

    for x in class_col:
        y_data[class_col.index(x)][list_of_classes.index(x)] = 1
        
    return x_data, y_data
