import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import os
import xml.etree.ElementTree as ET
import sklearn.model_selection
import numpy as np

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 268, 500, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def build_model(input_shape):
    print("creating model...")

    model = tf.keras.Sequential(name="LP_extraction")
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate=0.35))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model

def readXML(xmlname, path=None):
    if path == None:
        path = os.path.join('..','pics','train_dataset')
    xml = ET.parse(path + "/annotations/" + xmlname)
    filename = xml.find('filename').text
    bndbox_elements = xml.find('object').getchildren()[-1].getchildren()
    bndbox = []
    for element in bndbox_elements:
        bndbox.append(int(element.text))
    return (cv2.imread(path + "/images/" + filename), bndbox)

def load_data():
    print("loading data...")
    path = os.path.join('..','pics','train_dataset')
    files = os.listdir(path + '/annotations')
    input = []
    output = []
    for file in files:
        image, box = readXML(file)
        input.append(image)
        output.append(box)

    input_train, input_test, output_train, output_test = sklearn.model_selection.train_test_split(input, output,
                                                                                                  test_size=0.2,
                                                                                                  random_state=0)

    return input_train, input_test, output_train, output_test

def batch_generator(data_dir, image_paths, box, batch_size):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)

def train_model(model, input_train, input_test, output_train, output_test):
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}.h5',
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=False,
                                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=1.0e-4))

    #model.fit(input_train, output_train, epochs=8, callbacks=[checkpoint])
    print(model.evaluate(input_test, output_test))

if __name__ == "__main__":
    data = load_data()
    model = build_model(INPUT_SHAPE)
    #train_model(model, *data)
