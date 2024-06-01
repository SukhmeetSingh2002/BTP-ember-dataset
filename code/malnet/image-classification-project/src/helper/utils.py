import os
# import cv2
# import numpy as np
import pickle
import sys
# sys.path.append('..')

from data_loader import DataLoader
import concurrent.futures

data_loader = DataLoader()
data_dir = '../data'


def load_data(directory, split):
    return data_loader.load_data(directory=directory, split=split)



def pickle_store_dataset(train_dir,val_dir,test_dir):

    with concurrent.futures.ThreadPoolExecutor() as executor:
        train_future = executor.submit(load_data, train_dir, False)
        val_future = executor.submit(load_data, val_dir, False)
        test_future = executor.submit(load_data, test_dir, False)

        # Wait for all futures to complete
        train_data, train_labels = train_future.result()
        val_data, val_labels = val_future.result()
        test_data, test_labels = test_future.result()


    # Define a helper function to save data
    def save_pickle(file_name, data):
        with open(f'{data_dir}/{file_name}', 'wb') as f:
            pickle.dump(data, f)

    # Use concurrent.futures to save the data in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(save_pickle, 'train_data.pkl', train_data)
        executor.submit(save_pickle, 'train_labels.pkl', train_labels)
        executor.submit(save_pickle, 'test_data.pkl', test_data)
        executor.submit(save_pickle, 'test_labels.pkl', test_labels)
        executor.submit(save_pickle, 'val_data.pkl', val_data)
        executor.submit(save_pickle, 'val_labels.pkl', val_labels)

# Other utility functions can be added here


def load_from_pickle():
    # Define a helper function to load pickle data
    def load_pickle(file_name):
        with open(f'{data_dir}/{file_name}', 'rb') as f:
            return pickle.load(f)

    # Use concurrent.futures to load the data in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=70) as executor:
        train_data_future = executor.submit(load_pickle, 'train_data.pkl')
        train_labels_future = executor.submit(load_pickle, 'train_labels.pkl')
        test_data_future = executor.submit(load_pickle, 'test_data.pkl')
        test_labels_future = executor.submit(load_pickle, 'test_labels.pkl')
        val_data_future = executor.submit(load_pickle, 'val_data.pkl')
        val_labels_future = executor.submit(load_pickle, 'val_labels.pkl')

        # Wait for all futures to complete
        train_data = train_data_future.result()
        train_labels = train_labels_future.result()
        test_data = test_data_future.result()
        test_labels = test_labels_future.result()
        val_data = val_data_future.result()
        val_labels = val_labels_future.result()

    return train_data, train_labels, val_data, val_labels, test_data, test_labels
