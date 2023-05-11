import numpy as np
import sys
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback

print("Import successful!")

pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)


# defining model
def cnn(image_size, num_classes):
    classifier = Sequential()
    classifier.add(Conv2D(64, (5, 5), input_shape=image_size, activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(num_classes, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return classifier


image_size = (64, 64, 3)
datagen = ImageDataGenerator(rescale=1. / 255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             )
neuralnetwork_cnn = cnn(image_size, 150)
neuralnetwork_cnn.summary()

training_set = datagen.flow_from_directory("data/train",
                                           target_size=image_size[:2],
                                           batch_size=32,
                                           class_mode='categorical',
                                           color_mode='rgb'
                                           )

validation_set = datagen.flow_from_directory("data/validation",
                                             target_size=image_size[:2],
                                             batch_size=32,
                                             class_mode='categorical',
                                             color_mode='rgb'
                                             )

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
filepath = "model.h5"
ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
rlp = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1)

history = neuralnetwork_cnn.fit_generator(
    generator=training_set, validation_data=validation_set,
    callbacks=[es, ckpt, rlp], epochs=20,
)

# plot_model(neuralnetwork_cnn, show_shapes=True)  # just learned this nice new thing