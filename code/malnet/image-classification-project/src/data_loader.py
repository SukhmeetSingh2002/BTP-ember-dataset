import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def count_images(directory):
    num_images = 0
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for family_name in os.listdir(class_dir):
                family_dir = os.path.join(class_dir, family_name)
                for image_name in os.listdir(family_dir):
                    image_path = os.path.join(family_dir, image_name)
                    image = cv2.imread(image_path)
                    if image is not None and image.shape == (256, 256, 3):  # Check if image is 256x256 pixels with RGB
                        num_images += 1
    return num_images

def load_images_from_directory(directory):
    num_images = count_images(directory)
    images = np.empty((num_images, 256, 256, 3), dtype=np.uint8)
    labels = np.empty(num_images, dtype=object)

    print("Loading images from", directory)
    print("Classes found:", os.listdir(directory))

    image_index = 0
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for family_name in os.listdir(class_dir):
                family_dir = os.path.join(class_dir, family_name)
                print("Loading images from", family_dir, " "*(100-len(family_dir)), len(os.listdir(family_dir)), "images")
                
                for image_name in os.listdir(family_dir):
                    image_path = os.path.join(family_dir, image_name)
                    image = cv2.imread(image_path)
                    if image is not None and image.shape == (256, 256, 3):  # Check if image is 256x256 pixels with RGB
                        images[image_index] = image
                        labels[image_index] = class_name
                        image_index += 1

    print("Images loaded:", num_images)
    print("Labels loaded:", len(labels))
    return images, labels

def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        # Preprocess the image (e.g., resize, normalize, etc.)
        preprocessed_image = cv2.resize(image, (256, 256))
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
    # preprocessed_images = preprocess_images(images)

    if split:
        X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return images, encoded_labels

class DataLoader:
    def __init__(self, directory=None):
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
    