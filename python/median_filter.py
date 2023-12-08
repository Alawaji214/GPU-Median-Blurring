import cv2
import numpy as np
import time


# Function to apply median filter to a single channel with black edges
def apply_median_filter(input_channel, kernel_size):
    edge = kernel_size // 2
    # Initialize output with zeros (black edges)
    output = np.zeros_like(input_channel)
    for y in range(edge, input_channel.shape[0] - edge):
        for x in range(edge, input_channel.shape[1] - edge):
            neighbors = input_channel[y - edge:y + edge + 1, x - edge:x + edge + 1]
            output[y, x] = np.median(neighbors)
    return output

def main():
    # Read the image
    image = cv2.imread('../resources/sp_img_gray_noise_heavy.png', cv2.IMREAD_COLOR)

    # Split the image into its color channels
    channels = cv2.split(image)

    start_time = time.time()

    # Apply median filter to each channel
    filtered_channels = [apply_median_filter(ch, 5) for ch in channels]  # Kernel size is 5

    end_time = time.time()
    print(f"Filtering time: {end_time - start_time} seconds")
    
    # Merge the channels back
    filtered_image = cv2.merge(filtered_channels)

    # Save the images before and after filtering
    cv2.imwrite('before_py.jpg', image)
    cv2.imwrite('after_py.jpg', filtered_image)

if __name__ == "__main__":
    main()
