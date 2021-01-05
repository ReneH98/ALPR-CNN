import cv2
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import os
import xml.etree.ElementTree as ET
import sklearn.model_selection
import numpy as np
import matplotlib.image as mpimg

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 128, 128, 3
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
    return filename, bndbox

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
    input = np.array(input)
    output = np.array(output)
    input_train, input_test, output_train, output_test = sklearn.model_selection.train_test_split(input, output,
                                                                                                  test_size=0.2,
                                                                                                  random_state=0)
    return input_train, input_test, output_train, output_test

def load_image(data_dir, image_file):
    img = mpimg.imread(os.path.join(data_dir, os.path.basename(image_file)))
    if len(img.shape) > 2 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return img

def batch_generator(data_dir, image_paths, box, batch_size):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    boxes = np.empty([batch_size, 4])
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            img = image_paths[index]
            image = load_image(data_dir, img)
            bounding_box = box[index]

            images[i] = image
            boxes[i] = bounding_box
            i += 1
            if i == batch_size:
                break
        yield images, boxes

def train_model(model, input_train, input_test, output_train, output_test):
    checkpoint = tf.keras.callbacks.ModelCheckpoint('model-{epoch:03d}.h5',
                                                 monitor='val_loss',
                                                 verbose=0,
                                                 save_best_only=False,
                                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=1.0e-4))

    model.fit(batch_generator('../pics/train_dataset/images', input_train, output_train, 40),
              steps_per_epoch=2000,
              epochs=5,
              max_queue_size=10,
              validation_data=batch_generator('../pics/train_dataset/images', input_test, output_test, 40),
              validation_steps=len(input_test),
              callbacks=[checkpoint],
              verbose=1)

if __name__ == "__main__":
    data = load_data()
    model = build_model(INPUT_SHAPE)
    train_model(model, *data)
