from cnn_model import CNNModel
from data_loader import DataLoader
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Train or load a model.')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--load', action='store_true', help='Load the model')

args = parser.parse_args()

if not args.train and not args.load:
    parser.error('No action requested, add --train or --load')
    parser.print_help()

# Define the directories for image classification
data_dir = '../../../../../dataset/dataset_9010/dataset_9010/malimg_dataset/train'
test_dir = '../../../../../dataset/dataset_9010/dataset_9010/malimg_dataset/validation'
model_dir = '../models'
results_dir = '../results'

# Initialize the data loader
data_loader = DataLoader(data_dir)

# Load and preprocess the images
train_data, val_data, train_labels, val_labels = data_loader.load_data()
test_data, test_labels = data_loader.load_data(directory=test_dir, split=False)

original_train_labels = train_labels
original_test_labels = test_labels
original_val_labels = val_labels


# to categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
val_labels = to_categorical(val_labels)

print("10 labels:", train_labels[:10])

print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)
print("Test data shape:", test_data.shape)
print("Test labels shape:", test_labels.shape)
print("Test labels shape:", test_labels[0].shape)
print("Test labels shape:", test_labels[0])

input("Press Enter to continue...")

num_classes = train_labels.shape[1]
print("Number of classes:", np.unique(train_labels))
print("Number of classes:", np.unique(train_labels).shape)
print("Shape of train data:", train_data.shape)
print("Shape of train data:", train_data[0].shape)
print("Labels:", train_labels[0].shape,num_classes)

input("Press Enter to continue...")

# Initialize the CNN model
model = CNNModel(input_shape=train_data[0].shape, num_classes=num_classes)

# Train the model
# model.train(train_data, train_labels,num_epochs=10)

if args.train:
    model.train(train_data, train_labels, num_epochs=10, val_data=(val_data, val_labels))

# load the model
if args.load:
    model.load(model_dir)

print("Model loaded")
print("Test data shape:", test_data.shape)
print("Test labels shape:", test_labels.shape)
print("Test data shape:", test_labels[:10])


# Evaluate the model
accuracy, confusion_matrix, classification_report = model.evaluate(x_test=test_data,y_test=original_test_labels,verbose=1)

# Save the trained model
if args.train:
    model.save(model_dir)

# Print the accuracy
print(f"Accuracy: {accuracy}")