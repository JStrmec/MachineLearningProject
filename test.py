import keras_cnn as kcnn
import keras
from keras_cnn import keras_CNN as kcnn
import numpy as np
import data_processing as d
from tensorflow import keras

X, y = d.get_encoded_data()
X_train, X_test, y_train, y_test = d.splitData(X, y)


cnn1D = kcnn(3, 500)
new_X_train, new_X_test, new_y_train, new_y_test = cnn1D.reshape_and_chng_type(X_train, X_test, y_train, y_test)
model = cnn1D.getModel()
optimizer = keras.optimizers.Adam(learning_rate=0.005, decay=5e-4)

model.compile(optimizer, 'binary_crossentropy', metrics=[keras.metrics.Accuracy()])

history = model.fit(
    new_X_train,
    new_y_train,
    shuffle=True,
    verbose=1,
    epochs=150
)

# for metric in history.history:
#     cnn1D.plot(history.history, EPOCHS, metric)

print("Evaluate on test data")
pred_y = model.predict(new_X_test)
pred_y = pred_y.reshape(pred_y.shape[1], pred_y.shape[2]).astype(int)
pred_y = d.standarize_predictions(pred_y).squeeze()
count = successes = 0
for i in range(len(pred_y)):
    count += 1
    if np.array_equal(pred_y[i], y_test[i]):
        successes += 1


print("test acc:", float(successes) / count)