import os
import cv2
import numpy as np

def load_images_from_directory(directory):
    images = []
    labels = []
    class_names = os.listdir(directory)
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    images.append(image)
                    labels.append(class_name)
    return np.array(images), np.array(labels)

def preprocess_images(images):
    # Preprocess the images (e.g., resize, normalize, etc.)
    preprocessed_images = []
    for image in images:
        # Preprocess each image here
        preprocessed_image = image
        preprocessed_images.append(preprocessed_image)
    return np.array(preprocessed_images)

def evaluate_model(model, test_images, test_labels):
    # Evaluate the model on the test images
    # and return the accuracy
    accuracy = model.evaluate(test_images, test_labels)
    return accuracy

# Other utility functions can be added here