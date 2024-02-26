import random as rn
import tensorflow as tf 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.metrics import accuracy_score

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

def getANN_onehot_1(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def getANN_1(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def trainAnnModel(model, X_train, Y_train, X_val, Y_val, epochs = 10, batch_size = 32):
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val),verbose = 1)
    return history


def predictAnnModel(model,X_test,Y_test,threshold = 0.5):
    y_pred = model.predict(X_test)
    y_pred_binary = np.where(y_pred >threshold,1,0)
    y_pred_binary = np.array(y_pred_binary)
    print(accuracy_score(Y_test, y_pred_binary))
    metrics = model.evaluate(X_test, Y_test, verbose=1)
    print(*metrics)
    
    return y_pred, y_pred_binary