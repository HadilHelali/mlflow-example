import argparse
import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.config import experimental
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense

import shutil
import random


#gpus = experimental.list_physical_devices("GPU")
#experimental.set_memory_growth(gpus[0], True)

plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")
np.random.seed(42)

# create model_artifacts directory
model_artifacts_dir = "/tmp/model_artifacts"
Path(model_artifacts_dir).mkdir(exist_ok=True)

def remove_files_in_subdirs(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

# defining model
def cnn(image_size, num_classes):
    classifier = Sequential()
    classifier.add(Conv2D(64, (5, 5), input_shape=(100, 100, 1), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(num_classes, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return classifier


image_size = (100, 100, 1)
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

neuralnetwork_cnn = cnn(image_size, 151)
neuralnetwork_cnn.summary()

train_dir = "data/processed/train"
test_dir = "data/processed/test"
validation_dir = "data/processed/validation"


shutil.copytree("data/processed/PokemonData", train_dir)
shutil.copytree("data/processed/PokemonData", validation_dir)
shutil.copytree("data/processed/PokemonData", test_dir)


remove_files_in_subdirs(validation_dir)
remove_files_in_subdirs(test_dir)

def prep_data(pokemon, train_dir, _dir , nb):
  pop = os.listdir(train_dir+'/'+pokemon)
  length = len(pop)
  test_data= random.sample(pop, int(length * nb))
  print(test_data)
  for f in test_data:
    shutil.move(train_dir+'/'+pokemon+'/'+f, _dir+'/'+pokemon+'/')


for poke in os.listdir(train_dir):
  prep_data(poke, train_dir, validation_dir , 0.17)
for poke in os.listdir(train_dir):
  prep_data(poke, train_dir, test_dir , 0.13)



training_set = datagen.flow_from_directory(train_dir,
                                           target_size=image_size[:2],
                                           batch_size=32,
                                           class_mode='categorical',
                                           color_mode='grayscale'
                                           )

validation_set = datagen.flow_from_directory(validation_dir,
                                             target_size=image_size[:2],
                                             batch_size=32,
                                             class_mode='categorical',
                                             color_mode='grayscale'
                                             )

test_set = datagen.flow_from_directory(test_dir,
                                             target_size=image_size[:2],
                                             batch_size=32,
                                             class_mode='categorical',
                                             color_mode='grayscale'
                                             )

# main entry point
if __name__ == "__main__":

    print("MLFlow train ...")

    with mlflow.start_run(run_name="train") as run:
        mlflow.sklearn.autolog()
        run_id = run.info.run_id
        print("in start run ...")
        mlflow.set_tag("mlflow.runName", "train")
        print("after tag setting ...")

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)

        filepath = "model.h5"
        ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        rlp = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1)

        print("before history...")
    
        history = neuralnetwork_cnn.fit_generator(
                                generator=training_set,
                                validation_data=validation_set,
                                callbacks=[es, ckpt, rlp],
                                epochs=1
                                #steps_per_epoch=len(training_set),
                                #validation_steps=len(validation_set),
                                )
        # Access the metrics
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_accuracy = history.history['acc']
        val_accuracy = history.history['val_acc']

        print("saveing weights ...")
        neuralnetwork_cnn.save_weights("model/model.h5")

        # Log tracked parameters
        #mlflow.log_params(run_parameters)

        mlflow.sklearn.log_model(neuralnetwork_cnn, "model")

        # log charts
        # mlflow.log_artifacts(model_artifacts_dir)
        # Write metrics to file
        with open('metrics.txt', 'w') as outfile:
            outfile.write(f'\ntraining loss = {train_loss}.')
            outfile.write(f'\nvalidation loss = {val_loss}.')
            outfile.write(f'\nAccuracy = {train_accuracy}.')
            outfile.write(f'\nValidation Accuracy = {val_accuracy}.')

        os.listdir(f"mlruns/0/{run_id}/artifacts/model")