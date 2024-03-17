import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_images_from_directory(directory):
    images = []
    labels = []

    print("Loading images from", directory)
    print("Classes found:", os.listdir(directory))

    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)

        # print("Loading images from", class_dir, " "*10, len(os.listdir(class_dir)), "images") 
        # # there must be constant gap between class_dir and number of images
        print("Loading images from", class_dir, " "*(100-len(class_dir)), len(os.listdir(class_dir)), "images")


        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(class_name)
    
    print("Images loaded:", len(images))
    print("Labels loaded:", len(labels))
    return images, labels

def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        # Preprocess the image (e.g., resize, normalize, etc.)
        preprocessed_image = cv2.resize(image, (224, 224))
        preprocessed_image = preprocessed_image.astype('float32') / 255.0
        # grayscale_image
        # preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
        preprocessed_images.append(preprocessed_image)
    return np.array(preprocessed_images)

def load_data(directory, split):
    """
    Load and preprocess data from the given directory.

    Args:
        directory (str): The directory path containing the images.
        split (bool): Whether to split the data into training and testing sets.

    Returns:
        If split is True:
            X_train (array-like): The preprocessed training images.
            X_test (array-like): The preprocessed testing images.
            y_train (array-like): The encoded training labels.
            y_test (array-like): The encoded testing labels.
        If split is False:
            preprocessed_images (array-like): The preprocessed images.
            encoded_labels (array-like): The encoded labels.
    """
    images, labels = load_images_from_directory(directory)
    encoded_labels = LabelEncoder().fit_transform(labels)
    preprocessed_images = preprocess_images(images)

    if split:
        X_train, X_test, y_train, y_test = train_test_split(preprocessed_images, encoded_labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return preprocessed_images, encoded_labels

class DataLoader:
    def __init__(self, directory):
        self.directory = directory

    def load_data(self, directory=None, split=True):
            """
            Loads the data from the specified directory.

            Args:
                directory (str, optional): The directory path from which to load the data. If not provided, the default directory will be used.
                split (bool, optional): Whether to split the data into training and testing sets. Defaults to True.

            Returns:
                The loaded data.
                If split is True:
                    X_train (array-like): The preprocessed training images.
                    X_test (array-like): The preprocessed testing images.
                    y_train (array-like): The encoded training labels.
                    y_test (array-like): The encoded testing labels.
                If split is False:
                    preprocessed_images (array-like): The preprocessed images.
                    encoded_labels (array-like): The encoded labels.

            """
            if directory is not None:
                return load_data(directory, split)
            else:
                return load_data(self.directory, split)
    