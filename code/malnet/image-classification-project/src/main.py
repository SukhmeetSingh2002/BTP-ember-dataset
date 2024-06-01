from scipy import stats
from model import Model
import numpy as np
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse
import pickle
from helper.stats import stats_dataset
from helper.utils import pickle_store_dataset,load_from_pickle
import time 
# Create the parser
parser = argparse.ArgumentParser(description='Train or load a model.')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--pickle', action='store_true', help='Load the model from pickle')
parser.add_argument('--stats', action='store_true', help='Print the stats of the data')
parser.add_argument('--resnet', action='store_true', help='Use resnet model')
args = parser.parse_args()

train_dir = '../../../../../dataset/malnet/malnet-images-tiny/train'
test_dir = '../../../../../dataset/malnet/malnet-images-tiny/test'
val_dir =  '../../../../../dataset/malnet/malnet-images-tiny/val'
model_dir = './models'
results_dir = './results'



if not any([args.train, args.test, args.pickle,args.stats]):
    parser.error('No action requested,')
    parser.print_help()

from tensorflow import config as tf_config
print("GPUs Available: ", tf_config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf_config.list_physical_devices('GPU')))


if args.stats:
    stats_dataset(train_dir,'train_stats.txt')
    stats_dataset(test_dir,'test_stats.txt')
    stats_dataset(val_dir,'val_stats.txt')
    exit()
    
    
if args.pickle:
    start = time.time()
    pickle_store_dataset(train_dir,val_dir,test_dir)
    end = time.time()
    print("Data pickled")
    print("Time taken to pickle data:", end - start)
    exit()



start_time = time.time()
train_data, train_labels, val_data, val_labels, test_data, test_labels = load_from_pickle()
end_time = time.time()


print(type(val_labels),type(val_data))
print("Time taken to load data:", end_time - start_time)

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

# input("Press Enter to continue...")

num_classes = train_labels.shape[1]
print("Number of classes:", np.unique(train_labels))
print("Number of classes:", np.unique(train_labels).shape)
print("Shape of train data:", train_data.shape)
print("Shape of train data:", train_data[0].shape)
print("Labels:", train_labels[0].shape,num_classes)
if args.train or args.test:

    # input("Press Enter to continue...")

    # Initialize the CNN model
    model = Model(input_shape=train_data[0].shape, num_classes=num_classes)

    # Train the model
    # model.train(train_data, train_labels,num_epochs=10)

    if args.train:
        history=model.train(train_data, train_labels, num_epochs=10, val_data=(val_data, val_labels))
        pickle.dump(history, open(f'{results_dir}/history.pkl', 'wb'))

    # load the model
    if args.test:
        print("Loading the model")
        model.load(model_dir)
        
        print("Model loaded",history)

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
    
