import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import glob

from sklearn.model_selection import train_test_split


EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 3
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    cv_img = []
    labels = []
    # shaped_img = []
    
    # for img in glob.glob(os.path.sep.join([data_dir, "*","*.ppm"])): # when u use sep, make the inside of join a list []
    #     subdirname = os.path.basename(os.path.dirname(img))
    #     labels.append(subdirname)
    #     img = cv2.imread(img)
    #     cv_img.append(img)
    #     # img = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT))
    #     # cv_img.append(img)
    
    # for i in cv_img:
    #     i = cv2.resize(i, dsize=[IMG_WIDTH, IMG_HEIGHT])
    #     shaped_img.append(i)
    # # print(shaped_img)
    # return shaped_img, labels
    
    for root_path, _, files in os.walk(data_dir):
        for file in files:
            if not file.startswith('.'):
                img = cv2.imread(os.path.join(root_path, file))
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                cv_img.append(img)
                # labels.append(int(os.path.basename(root_path)))
                labels.append(int(os.path.basename(root_path.rstrip(os.sep)))) # platform-independent
    return (cv_img, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # model = tf.keras.models.Sequential() # Create a neural network. keras is an API. Sequential neurak network means one layer after another 
    # model.add(tf.keras.layers.Dense(8, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation="relu")) # Add a hidden layer with 8 units
    # model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')) # Add output layer with 3 units
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D( # First concultional layer learning 32 filters using a 3 by 3 kernel a.k.a filter
        64, (3,3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        #add max-pooling layer with 2 by 2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),


        tf.keras.layers.Conv2D( # 2nd concultional layer learning 32 filters using a 3 by 3 kernel a.k.a filter
        64, (3,3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.Flatten(), # flattening the units

        # add hidden layer with 128 units and dropout of 50%
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # add an output layer with output units, one for each category
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])


    # train neural network
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
