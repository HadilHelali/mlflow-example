import mlflow
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from io import BytesIO
from PIL import Image
import svglib.svglib as svglib
import pickle





# Define constants
IMG_SIZE = 100

# Define preprocessing functions
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    ## resizing the image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    ## converting the image's color space to Gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## normalizing the data
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_label(label):
    return label

# Define data loading function
def load_data():
    data = []
    labels = []
    for subdir, _, files in os.walk('data/raw/PokemonData'):
        for file in files:
            file_extension = file.split('.')[-1].lower()
            if(file_extension == "jpg" or file_extension == "png" or file_extension == "jpeg"):
                img_path = os.path.join(subdir, file)
                label = os.path.basename(subdir)

                print(img_path + " : " + label)
                # preprocessing image + label
                data.append(preprocess_image(img_path))
                labels.append(preprocess_label(label))
    return np.array(data), np.array(labels)

# Define function to add noise to images
def add_noise(images, std_dev):
    print("--------- here we go we're adding some noise -----------")
    noise = np.random.normal(loc=0.0, scale=std_dev, size=images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0.0, 1.0)


# Define function to remove background from images
def remove_background(images):
    print("----- get started to remove the background -----")
    gray_images = images
    _, mask = cv2.threshold(gray_images, 10, 255, cv2.THRESH_BINARY)
    masked_images = cv2.bitwise_and(images, mask)
    return masked_images

# Define function to simulate performance degradation
def simulate_degradation(images, prob):
    degraded_images = np.copy(images)
    num_pixels = images.shape[1] * images.shape[2]
    for i in range(images.shape[0]):
        pixels_to_degrade = np.random.choice(num_pixels, size=int(prob*num_pixels), replace=False)
        degraded_images[i, pixels_to_degrade // images.shape[1], pixels_to_degrade % images.shape[1], :] = 0.0
    return degraded_images

if __name__ == "__main__":

    with mlflow.start_run(run_name="load_raw_data") as run:

        mlflow.set_tag("mlflow.runName", "load_raw_data")

        # load data and labels
        data, labels = load_data()

        data = remove_background(data)
        data = add_noise(data,0.1)

        data = list(zip(data, labels))

        # `data` is a list of tuples containing the processed images and their labels
        # [(image1, label1), (image2, label2), ...]
        with open('data/processed/images.pickle', 'wb') as f:
            pickle.dump(data, f)



#         with open('data.pickle', 'rb') as f:
#             data = pickle.load(f)

        print(data[0])

