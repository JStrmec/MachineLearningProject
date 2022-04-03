# Building a Convolutional Neural Network to classifiy computer 
# attacks as types of malware
from sklearn.model_selection import train_test_split
import data_processing as d
X, y = d.get_encoded_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)