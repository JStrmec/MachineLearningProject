import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow import keras
import matplotlib.pyplot as plt

LAYERS = 16 # arbitrary number for now
EPOCHS = 1000 # arbitrary

# TODO: define all the testing data

def getModel():
    # all three layers are arbitrarily chosen, don't know input shape
    return keras.Sequential(
        [
            layers.Dense(LAYERS, activation='relu', name='layer1'),
            layers.Dense(LAYERS / 2, activation='relu', name='layer2'),
            layers.Dense(LAYERS / 4, activation='relu', name='layer3')
        ])

def plot(history, epochs):
    loss_train = history.history['acc']
    loss_val = history.history['val_acc']
    epochs = range(1,epochs)
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    model = getModel()
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer, 'mse', metrics=[keras.metrics.Accuracy(), 
                                                keras.metrics.MeanSquaredError(),
                                                keras.metrics.Precision(), 
                                                keras.metrics.Recall(),
                                                keras.metrics.RootMeanSquaredError()])
    
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=EPOCHS,
        validation_data=(x_val, y_val)
    )
    
    plot(history, EPOCHS)
    
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)
    
    