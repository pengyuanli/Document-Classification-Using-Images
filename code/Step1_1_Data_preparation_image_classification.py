# Deep learning for image classification

# Use pre-trained VGG for image classification
# Results not very well need further finetune

# {'3D': 0, 'BarChart': 1, 'Fluorescence': 2, 'Gel': 3, 'LineChart': 4, 'Microscopy': 5, 'OtherGraph': 6, 'Plate': 7, 'Sequence': 8, 'Text': 9, 'picture': 10, 'unknown': 11}

# No class weight acc: 0.8700 - val_loss: 0.7249 - val_acc: 0.7775
# Having class weight acc: 0.8710 - val_loss: 0.8222 - val_acc: 0.7609

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from collections import Counter
import os

classweight = True
date = '_0305'
data_source = '_on_Luis_data'
model_name = 'VGG'
pretrain = False

if classweight:
    suffix = model_name + data_source + date + '_classweight'
else:
    suffix = model_name + data_source + date+'_no_classweight'

if pretrain:
    suffix = suffix + '_pretrain'

print(suffix)

# Data loading # Data preparation
data_path = '/media/pengyuan/Research/Research/PPI/David_PPI_image/data'
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
    )

train_it = train_datagen.flow_from_directory(
    data_path,
    class_mode='categorical',
    batch_size= batch_size,
    target_size=(150, 150),
    subset='training')

validate_it = train_datagen.flow_from_directory(
    data_path,
    class_mode='categorical',
    batch_size= batch_size,
    target_size=(150, 150),
    subset='validation')

# Calculate class weight
counter = Counter(train_it.classes)
max_val = float(max(counter.values()))
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

# Define model
prev_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3)) #
print('Model loaded.')
if pretrain:
    for layer in prev_model.layers:
        layer.trainable = False

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(prev_model)
top_model.add(Flatten(input_shape=prev_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.1))
top_model.add(Dense(13, activation='softmax'))



# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
top_model.compile(loss='categorical_crossentropy',
              optimizer='adam', # change to adam, sgd
              metrics=['accuracy'])

top_model.fit_generator(
        train_it,
        steps_per_epoch=5000 // batch_size,
        epochs=50,
        class_weight=class_weights,
        validation_data = validate_it,
        validation_steps=1000 // batch_size)

top_model.save_weights('/media/pengyuan/Research/DOC/Code/temp/' + suffix + '.h5')  # a