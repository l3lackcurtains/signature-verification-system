import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import cv2
import operator
import preprocessing


def main():

    train_dir = os.fsencode('./Dataset/custom2')
    test_dir = os.fsencode('./Dataset/custom2-test')

    training_data = []
    training_label = []
    testing_data = []
    testing_label = []

    # Get Training Data
    for filename in os.listdir(train_dir):

        filename_decoded = filename.decode('utf-8')
        train_dir_decoded = train_dir.decode('utf-8')

        impath = train_dir_decoded+'/'+filename_decoded

        imgs = np.array(preprocessing.prepare(
            impath, filename_decoded, augment=True))
        
        for data in imgs:
            # Reshape with channel
            data = data.reshape(1, 32, 32)
            # normalize image
            data = data.astype('float32')
            data = data / 255

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

        # Reshape with channel
        data = data.reshape(1, 32, 32)

        # normalize image
        data = data.astype('float32')
        data = data / 255
        
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



    # Create Training and Testing Sets
    training_data = np.array(training_data)
    training_label = np.array(training_label)
    testing_data = np.array(testing_data)
    testing_label = np.array(testing_label)

    # input shape of image 
    input_shape= (1, 32, 32)

    # creating a model
    model = keras.Sequential()

    # convolution Layer 1
    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), data_format='channels_first',
                    activation='relu',
                    input_shape=input_shape))
    # convolution Layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    # convolution Layer 3
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    # fully connected layer 1
    model.add(keras.layers.Dense(256, activation='sigmoid'))
    model.add(keras.layers.Dropout(0.5))
    # fully connected layer 2
    model.add(keras.layers.Dense(32, activation='sigmoid'))
    # output layer
    model.add(keras.layers.Dense(6, activation='softmax'))



    # compiling the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Start Training
    model.fit(training_data, training_label, epochs=120)
    test_loss, test_acc = model.evaluate(testing_data, testing_label)

    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    predictions = model.predict(testing_data)

    print(predictions)

main()
