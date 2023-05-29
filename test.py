import cv2
import mlflow
import numpy as np
from preprocess import preprocess_image,add_noise,remove_background,simulate_degradation
IMG_SIZE=100

def test_preprocess_image():
    # Define the input image path
    img_path = 'test_image.jpg'

    # Create a dummy image
    img = np.random.randint(0, 255, (200, 200, 3)).astype(np.uint8)
    cv2.imwrite(img_path, img)

    # Preprocess the image
    preprocessed_img = preprocess_image(img_path)

    # Assert that the output image has the expected shape
    assert preprocessed_img.shape == (IMG_SIZE, IMG_SIZE)
    
    # Assert that the output image is grayscale
    assert len(preprocessed_img.shape) == 2

def test_add_noise():
    # Create a dummy set of images
    images = np.random.rand(10, 100, 100)

    # Test adding noise with standard deviation of 0
    noisy_images = add_noise(images, 0.0)
    assert np.allclose(noisy_images, images)

    # Test adding noise with standard deviation of 0.1
    noisy_images = add_noise(images, 0.1)
    assert noisy_images.shape == images.shape
  
    assert np.max(noisy_images) <= 1.0
    assert np.min(noisy_images) >= 0.0

def test_image_processing():
    # Create a test input image with some low-intensity pixels
    input_image = np.zeros((100, 100), dtype=np.uint8)
    input_image[30:40, 30:40] = 5
    
    # Apply the image processing code
    masked_images = remove_background(input_image)
    
    # Check that the resulting image only contains high-intensity pixels
    assert np.all(masked_images[30:40, 30:40] == 0)


def test_simulate_degradation():
    input_image = np.random.randint(2, 256, size=(5, 10, 10, 3), dtype=np.uint8)

    # Print the shape of the resulting image
    print(input_image.shape)
    input_image[input_image == 0] = 1
    degraded_images = simulate_degradation(input_image, prob=0.5)

    # Check that the degraded image has some pixels set to 0
    assert np.any(degraded_images[2:4, 2:4, 2:4, :] == 0.0)

    # Check that the degraded image has the expected number of 0 pixels
    num_pixels = input_image.shape[1] * input_image.shape[2] * input_image.shape[3]
    expected_num_zeros = int(0.5 * num_pixels * input_image.shape[0])
    actual_num_zeros = np.count_nonzero(degraded_images == 0.0)
    assert actual_num_zeros == expected_num_zeros
  

# main entry point
if __name__ == "__main__":

    print("MLFlow tests ...")

    with mlflow.start_run(run_name="test") as run:
        mlflow.set_tag("mlflow.runName", "test")