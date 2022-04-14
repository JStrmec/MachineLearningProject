# Building a Convolutional Neural Network to classifiy computer 
# attacks as types of malware
import torch.nn as nn
import torch
import data_processing as d
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

X, y = d.get_encoded_data()
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

#####MESSING AROUND#############
# Importing libraries
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras import layers
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# Building the CNN Model
model = Sequential()
model.add(layers.Conv1D(128, 5, activation='relu',  input_shape=[1,7966], padding='same'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Flatten())
model.add(layers.Dense(26, activation='relu'))
model.add(layers.Dense(13, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

# Fitting the data onto model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Getting score metrics from our model
scores = model.evaluate(X_test, y_test, verbose=0)
# Displays the accuracy of correct sentiment prediction over test data
print("Accuracy: %.2f%%" % (scores[1]*100))