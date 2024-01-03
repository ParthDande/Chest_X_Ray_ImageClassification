# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:04:39 2023

@author: Parth
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
train_dir = r"chest_xray\chest_xray\train"
test_dir =r"chest_xray\chest_xray\val"

# Set up data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Use rescaling for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Define batch size and image dimensions
batch_size = 32
img_height = 150
img_width = 150

# Generate training data from the directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    classes=['NORMAL', 'PNEUMONIA',]  # Specify the correct class names
)

print("Validation Generator Class Indices:", validation_generator.class_indices)
print("Validation Generator File List:", validation_generator.filenames)


# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (normal or pneumonia)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=3,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Assuming 'model' is your trained model
model.save(r'C:\CODING\DataScience\Neural_Networks\Chest_Xray_Classifiaction')
