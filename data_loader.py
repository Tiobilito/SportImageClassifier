# data_loader.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(dataset_dir, target_size=(128, 128), batch_size=32, validation_split=0.2):
    # Generador para entrenamiento con aumento de datos y normalización
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    valid_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, valid_generator
