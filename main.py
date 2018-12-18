import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import cv2
import operator
import preprocessing


def main():

    train_dir = os.fsencode('./Dataset/custom2')
    test_dir = os.fsencode('./Dataset/custom-test')

    training_data = []
    training_label = []
    testing_data = []
    testing_label = []

    # Get Training Data
    for filename in os.listdir(train_dir):

        filename_decoded = filename.decode('utf-8')
        train_dir_decoded = train_dir.decode('utf-8')

        impath = train_dir_decoded+'/'+filename_decoded

        data = np.array(preprocessing.prepare(
            impath, filename_decoded))
        data = np.reshape(data, (1024, 1))

        # TODO: Make it short and simple
        result = [0, 0, 0, 0, 0, 0]
        if operator.contains(filename_decoded, 'p1'):
            result = [1, 0, 0, 0, 0, 0]
        elif operator.contains(filename_decoded, 'p2'):
            result = [0, 1, 0, 0, 0, 0]
        elif operator.contains(filename_decoded, 'p3'):
            result = [0, 0, 1, 0, 0, 0]
        elif operator.contains(filename_decoded, 'p4'):
            result = [0, 0, 0, 1, 0, 0]
        elif operator.contains(filename_decoded, 'p5'):
            result = [0, 0, 0, 0, 1, 0]
        elif operator.contains(filename_decoded, 'p6'):
            result = [0, 0, 0, 0, 0, 1]

        result = np.array(result)
        training_data.append(data)
        training_label.append(result)

    # Get Test Data
    for filename in os.listdir(test_dir):

        filename_decoded = filename.decode('utf-8')
        test_dir_decoded = test_dir.decode('utf-8')

        impath = test_dir_decoded+'/'+filename_decoded
        data = np.array(preprocessing.prepare(
            impath, filename_decoded))
        data = np.reshape(data, (1024, 1))
        # TODO: Make it short and simple
        result = [0, 0, 0, 0, 0, 0]
        if operator.contains(filename_decoded, 't1'):
            result = [1, 0, 0, 0, 0, 0]
        elif operator.contains(filename_decoded, 't2'):
            result = [0, 1, 0, 0, 0, 0]
        elif operator.contains(filename_decoded, 't3'):
            result = [0, 0, 1, 0, 0, 0]
        elif operator.contains(filename_decoded, 't4'):
            result = [0, 0, 0, 1, 0, 0]
        elif operator.contains(filename_decoded, 't5'):
            result = [0, 0, 0, 0, 1, 0]
        elif operator.contains(filename_decoded, 't6'):
            result = [0, 0, 0, 0, 0, 1]
        result = np.array(result)
        testing_data.append(data)
        testing_label.append(result)

    training_data = np.array(training_data)
    training_label = np.array(training_label)
    testing_data = np.array(testing_data)
    testing_label = np.array(testing_label)

    # creating a model
    model = keras.Sequential()

    model.add(keras.layers.Flatten(input_shape=(1024, 1)))
    model.add(keras.layers.Dense(256, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(128, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(256, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(6, activation=tf.nn.softmax))

    # compiling the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Start Training
    model.fit(training_data, training_label, epochs=25)
    test_loss, test_acc = model.evaluate(testing_data, testing_label)

    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    predictions = model.predict(testing_data)

    print(predictions)


main()
