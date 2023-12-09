import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import argparse


# Function to apply median filter to a single channel with black edges
def apply_median_filter(input_channel, kernel_size):
    edge = kernel_size // 2
    # Initialize output with zeros (black edges)
    output = np.zeros_like(input_channel)
    # start_time = time.time()
    for x in range(edge, input_channel.shape[1] - edge):
        for y in range(edge, input_channel.shape[0] - edge):
            neighbors = input_channel[y - edge:y + edge + 1, x - edge:x + edge + 1]
            output[y, x] = np.median(neighbors)
    
    # end_time = time.time()
    # print(f"loop time: {end_time - start_time} seconds")
    return output

def main():
    parser = argparse.ArgumentParser(description='Apply median filter to an image.')
    parser.add_argument('image_file', help='Path to the image file')

    args = parser.parse_args()

    # Read the image using the provided file path
    image_file_name = args.image_file
    image = cv2.imread(image_file_name, cv2.IMREAD_COLOR)
    # Read the image
    # image = cv2.imread('../resources/noise_intro_2.jpg', cv2.IMREAD_COLOR)

    # Split the image into its color channels
    channels = cv2.split(image)

    start_time = time.time()

    # Apply median filter to each channel
    with ProcessPoolExecutor() as executor:
      filtered_channels = list(executor.map(apply_median_filter, channels, [5] * len(channels)))
    # filtered_channels = [apply_median_filter(ch, 5) for ch in channels]  # Kernel size is 5
    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(apply_median_filter, ch, 5) for ch in channels]
    #     filtered_channels = [future.result() for future in futures] 
    end_time = time.time()
    print(f"Filtering time: {end_time - start_time} seconds")
    
    # Merge the channels back
    filtered_image = cv2.merge(filtered_channels)

    # Save the images before and after filtering
    cv2.imwrite('before_py_multi.jpg', image)
    cv2.imwrite('after_py_multi.jpg', filtered_image)

if __name__ == "__main__":
    main()
