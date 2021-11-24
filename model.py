import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Helper libraries
import numpy as np
import pathlib

import requests


class_names = {0: { 'id':'181008073', 'name':'African Elephant'}, 
               1: { 'id':'15954', 'name':'Leopard'}, 
               2: { 'id':'899', 'name': 'Arctic Fox'}, 
               3: { 'id':'15933', 'name':'Chimpanzee'}, 
               4: {'id': '17975', 'name': 'Bornean Orangutan'}}

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
NUM_CLASSES = len(class_names)
VALIDATION_SPLIT = 0.2
EPOCHS = 15

MODEL_DIR = pathlib.Path('./model/1')
DATA_DIR = pathlib.Path('./data')

API_TOKEN = 'TOKEN'


# Definition of a model
def create_model():

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(IMG_HEIGHT,
                                                                      IMG_WIDTH,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])

    return model


def train_model(model):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    return model


def save_model(model):
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

def load_model():
    restored_model = tf.keras.models.load_model(
        MODEL_DIR, custom_objects=None, compile=True, options=None
    )
    return restored_model


def get_prediction(img_obj):

    model = load_model()

    img = keras.preprocessing.image.load_img(
        img_obj, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = model.predict(img_array)

    score = tf.nn.softmax(predictions[0])

    class_obj = class_names[np.argmax(score)]
    accuracy = 100 * np.max(score)

    if (accuracy > 70):
        return (class_obj, accuracy)
    else:
        return ('Unknown', 100)


def get_result(img_obj):
    (class_obj, acc) = get_prediction(img_obj)
    query = 'https://apiv3.iucnredlist.org/api/v3/species/narrative/id/%s?token=%s' % (class_obj['id'], API_TOKEN)

    response = requests.get(query)

    response_json = response.json()['result'][0]

    result = "This image most likely belongs to %s with a %d percent confidence.\n \n *Population Trend* \n %s \n \n *Threats* \n %s" % (
            class_obj['name'], acc, response_json['populationtrend'], response_json['threats'][0:3500])

    return result


