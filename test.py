import keras_cnn as kcnn
import keras
from keras_cnn import keras_CNN as kcnn
import numpy as np
import data_processing as d
from tensorflow import keras
from sklearn.metrics import accuracy_score

X, y = d.get_encoded_data()
X_train, X_test, y_train, y_test = d.splitData(X, y)


cnn1D = kcnn(7, 500)
new_X_train, new_X_test, new_y_train, new_y_test = cnn1D.reshape_and_chng_type(X_train, X_test, y_train, y_test)
model = cnn1D.getModel()
optimizer = keras.optimizers.Nadam(learning_rate=1e-3, decay=5e-4)

model.compile(optimizer, 'mse', metrics=[keras.metrics.Accuracy()])

history = model.fit(
    new_X_train,
    new_y_train,
    batch_size=500,
    verbose=1,
    epochs=150
)

# for metric in history.history:
#     cnn1D.plot(history.history, EPOCHS, metric)

print("Evaluate on test data")
results = model.predict(new_X_test, batch_size=64)
results = results.reshape(results.shape[1], results.shape[2]).astype(int)

print("test acc:", accuracy_score(results, y_test))