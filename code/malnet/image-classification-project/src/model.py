import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
from keras.applications import ResNet50

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def create_model(input_shape, num_classes):
    # model = tf.keras.Sequential([
    #     layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(num_classes, activation='softmax')
    # ])

    if len(input_shape) == 2:
        input_shape = (input_shape[0],input_shape[1], 1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Flatten())

    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2 ))
    model.add(Dense(num_classes,activation='softmax'))

    return model

def create_model2(input_shape, num_classes):
    # input_shape = (input_shape[0],input_shape[1], 1)
    
    # model = Sequential()
    # model.add(Conv2D(128, kernel_size=(2,2), input_shape=input_shape))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
    # #Flatten
    # model.add(Flatten())
    

    # # fully connected layer
    # model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.2))

    # model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.2))

    # # output layer
    # model.add(Dense(1, activation='sigmoid'))

    # # Compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # # model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
    # # model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

    # # Print the model summary
    # print(model.summary())
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(2,2), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
    #Flatten
    model.add(Flatten())

    # fully connected layer
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))

    # output layer
    model.add(Dense(num_classes, activation='softmax'))

    return model
    
def create_model_rsnet(input_shape, num_classes):
    #use resnet50
    model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape,classes=num_classes)
    model.trainable = True
    x = keras.layers.GlobalAveragePooling2D()(model.output)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[model.input], outputs=[output])

    return model


def train_model(model, train_data, train_labels, num_epochs, val_data=None):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Summary of the model\n", model.summary())
    
    # history = model.fit(train_data, validation_data=val_data, epochs=num_epochs)
    # history = model.fit(train_data, train_labels, epochs=num_epochs)
    history = model.fit(train_data, train_labels, epochs=num_epochs, validation_data=val_data, batch_size=1024)
    
    return history

def predict(model, test_data):
    predictions = model.predict(test_data)
    return predictions

class Model:
    def __init__(self, input_shape, num_classes):
        self.model = create_model_rsnet(input_shape, num_classes)
        self.history = None
    
    def train(self, x_train, y_train, num_epochs, val_data=None):
            """
            Trains the model using the given training data.

            Args:
                x_train (numpy.ndarray): The input training data.
                y_train (numpy.ndarray): The target training data.
                num_epochs (int): The number of epochs to train the model.
                val_data (tuple, optional): Validation data as a tuple of input and target data. Defaults to None.

            Returns:
                The trained model.
            """
            print("Training the model")
            print("x_train shape:", x_train.shape)
            print("y_train shape:", y_train.shape)

            # 1st row of both
            print("x_train[0]:", x_train[0])
            print("y_train[0]:", y_train[0])

            self.history = train_model(self.model, x_train, y_train, num_epochs, val_data)
            return self.history

    def evaluate(self, x_test, y_test, verbose=1):
        """
        Evaluate the model's performance on the test data.

        Args:
            x_test (numpy.ndarray): The input test data.
            y_test (numpy.ndarray): The target test data.
            verbose (int, optional): Verbosity mode. Set to 1 to print the evaluation results. Defaults to 1.

        Returns:
            tuple: A tuple containing the accuracy score, confusion matrix, and classification report.

        """
        predictions = predict(self.model, x_test)
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(predictions, y_test)

        if verbose:
            print("Accuracy:", accuracy)
            print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
            print("Classification report:\n", classification_report(y_test, predictions))

        return accuracy, confusion_matrix(y_test, predictions), classification_report(y_test, predictions)
    
    def predict(self, test_data):
        return predict(self.model, test_data)
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

