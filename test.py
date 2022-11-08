INPUTSHAPE = (512,512,3)
DROPOUT_RATE = 0.2
NUMBER_OF_CLASSES = 14
height = 512
width = 512
batch_size = 4
epochs = 10

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

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession


# def fix_gpu():
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = InteractiveSession(config=config)


# fix_gpu()

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, ... up to  7
# Higher the number, the more complex the model is. and the larger resolutions it  can handle, but  the more GPU memory it will need
# loading pretrained conv base model
#input_shape is (height, width, number of channels) for images
conv_base = EfficientNetB6(weights="imagenet", include_top=False, input_shape=INPUTSHAPE)

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
#avoid overfitting
model.add(layers.Dropout(rate=DROPOUT_RATE, name="dropout_out"))
# Set NUMBER_OF_CLASSES to the number of your final predictions.
model.add(layers.Dense(NUMBER_OF_CLASSES, activation="softmax", name="fc_out"))
conv_base.trainable = False

TRAIN_IMAGES_PATH = './vinbigdata/images/train' #12000
VAL_IMAGES_PATH = './vinbigdata/images/val' #3000
External_DIR = './vinbigdata-512-image-dataset/train' # 15000
os.makedirs(TRAIN_IMAGES_PATH, exist_ok = True)
os.makedirs(VAL_IMAGES_PATH, exist_ok = True)

classes = ['Aortic enlargement','Atelectasis','Calcification',
'Cardiomegaly','Consolidation','ILD','Infiltration',
'Lung Opacity','No finding','Other lesion',
'Pleural effusion','Pleural thickening','Pneumothorax',
'Pulmonary fibrosis']

# Create directories for each class.
for class_id in [x for x in range(len(classes))]:
    os.makedirs(os.path.join(TRAIN_IMAGES_PATH, str(class_id)), exist_ok = True)
    os.makedirs(os.path.join(VAL_IMAGES_PATH, str(class_id)), exist_ok = True)

Input_dir = './vinbigdata-512-image-dataset/train'
def preproccess_data(df, images_path):
    for column, row in tqdm(df.iterrows(), total=len(df)):
        class_id = row['class_id']
        shutil.copy(os.path.join(Input_dir, f"{row['image_id']}.png"), os.path.join(images_path, str(class_id)))
df = pd.read_csv('./vinbigdata-512-image-dataset/train.csv')
df.head()
#Split the dataset into 80% training and 20% validation
df_train, df_valid = model_selection.train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

#Run below 2 linse for only the one time 
#preproccess_data(df_train, TRAIN_IMAGES_PATH)
#preproccess_data(df_valid, VAL_IMAGES_PATH)

# I love the  ImageDataGenerator class, it allows us to specifiy whatever augmentations we want so easily...
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
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


model.summary()
model.save_weights('./model_saved/test.h5')

exit()
### Total classes
['Aortic enlargement','Atelectasis','Calcification',
'Cardiomegaly','Consolidation','ILD','Infiltration',
'Lung Opacity','No finding','Nodule/Mass','Other lesion',
'Pleural effusion','Pleural thickening','Pneumothorax',
'Pulmonary fibrosis']

