import os
import warnings
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
from skimage import color
from skimage import io
from PIL import Image
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import ImageFile
import sys
from matplotlib import pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

image_path = 'data'
models_path = 'models/saved_model.hdf5'
rgb = False
imageSize = 224

gestures = {'a_':'A',
            'b_':'B',
            'c_':'C',
            'd_':'D',
            'e_':'E',
            'f_':'F',
            'g_':'G',
            'h_':'H',
            'i_':'I',
            'k_':'K',
            'l_':'L',
            'm_':'M',
            'n_':'N',
            'o_':'O',
            'p_':'P',
            'q_':'Q',
            'r_':'R',
            's_':'S',
            't_':'T',
            'u_':'U',
            'v_':'V',
            'w_':'W',
            'x_':'X',
            'y_':'Y'
            }

gestures_map = {'A':0,
            'B':1,
            'C':2,
            'D':3,
            'E':4,
            'F':5,
            'G':6,
            'H':7,
            'I':8,
            'K':9,
            'L':10,
            'M':11,
            'N':12,
            'O':13,
            'P':14,
            'Q':15,
            'R':16,
            'S':17,
            'T':18,
            'U':19,
            'V':20,
            'W':21,
            'X':22,
            'Y':23
                }


gesture_names = {0:'A',
            1:'B',
            2:'C',
            3:'D',
            4:'E',
            5:'F',
            6:'G',
            7:'H',
            8:'I',
            9:'K',
            10:'L',
            11:'M',
            12:'N',
            13:'O',
            14:'P',
            15:'Q',
            16:'R',
            17:'S',
            18:'T',
            19:'U',
            20:'V',
            21:'W',
            22:'X',
            23:'Y'
                }


# Data generators
train_datagen = ImageDataGenerator(
    rescale= 1. / 255,
    rotation_range= 25,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    shear_range= 0.2,
    zoom_range= 0.2,
    horizontal_flip=True,
    fill_mode='nearest')
# Data flow
train_generator = train_datagen.flow_from_directory(
    directory="data/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
valid_datagen = ImageDataGenerator(rescale=1. / 255)
valid_generator = valid_datagen.flow_from_directory(
    directory="data/valid/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    directory="data/test/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)
# Dat cac checkpoint de luu lai model tot nhat
model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc',
                               min_delta=0,
                               patience=10,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)

# Khoi tao model
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
optimizer1 = optimizers.Adam()
base_model = model1

# Them cac lop ben tren
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dense(128, activation='relu', name='fc2a')(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', name='fc4')(x)

predictions = Dense(24, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Dong bang cac lop duoi, chi train lop ben tren minh them vao
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

n_train_steps = train_generator.n // train_generator.batch_size
n_valid_steps = valid_generator.n // valid_generator.batch_size
history=model.fit_generator(generator=train_generator,
                    steps_per_epoch=n_train_steps,
                    validation_data=valid_generator,
                    validation_steps=n_valid_steps,
                    epochs=100,
                    verbose=1,
                    callbacks=[early_stopping, model_checkpoint])
# Luu model da train ra file
model.save('models/mymodel.h5')


  
# Plot diagnostic learning curves
def summarize_diagnostics(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# Evaluate model
_,acc = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=1)

print('> %.3f' % (acc * 100.0))

# Learning curves
summarize_diagnostics(history)