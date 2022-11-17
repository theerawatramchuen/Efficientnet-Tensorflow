INPUTSHAPE = (512,512,3)
DROPOUT_RATE = 0.2
NUMBER_OF_CLASSES = 2
height = 512
width = 512
batch_size = 4
epochs = 600

from tensorflow.keras.applications import * #Efficient Net included here
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from tensorflow.keras import optimizers
import tensorflow as tf
#Use this to check if the GPU is configured correctly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, ... up to  7
# Higher the number, the more complex the model is. and the larger resolutions it  can handle, but  the more GPU memory it will need
# loading pretrained conv base model
#input_shape is (height, width, number of channels) for images
conv_base = EfficientNetB6(weights="imagenet", include_top=False, input_shape=INPUTSHAPE)

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
#model.add(layers.Dropout(rate=DROPOUT_RATE, name="dropout_out"))
model.add(layers.Dense(NUMBER_OF_CLASSES, activation="softmax", name="fc_out"))
conv_base.trainable = False

TRAIN_IMAGES_PATH = './dataset_saw/train' #12000
VAL_IMAGES_PATH = './dataset_saw/test' #3000

classes = ['bad','good']

# I love the  ImageDataGenerator class, it allows us to specifiy whatever augmentations we want so easily...
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
#    rotation_range=40,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True,
#    fill_mode="nearest",
)
# Note that the validation data should not be augmented!
#and a very important step is to normalise the images through  rescaling
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    TRAIN_IMAGES_PATH,
    # All images will be resized to target height and width.
    target_size=(height, width),
    batch_size=batch_size,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode="categorical",
)
validation_generator = test_datagen.flow_from_directory(
    VAL_IMAGES_PATH,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode="categorical",
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=["acc"],
)

model.summary()

#Load Saved Model for continue training
#model = tf.keras.models.load_model('./model_saved/saw.h5')

#Train the model:

history = model.fit_generator(
    train_generator,
    steps_per_epoch=batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=batch_size,
    verbose=1,
    use_multiprocessing=True,
    workers=4,
)

model.save('./model_saved/saw.h5')
############################################################
exit()
