import cv2
import numpy as np
from numba import cuda, uint8
import time
import argparse

# Define maximum kernel size (must be odd)
MAX_KERNEL_SIZE = 25  # Example for 5x5 kernel
assert MAX_KERNEL_SIZE % 2 == 1, "Kernel size must be odd"

@cuda.jit
def bitonic_sort(neighbors, count):
    # Bitonic sort implementation for sorting 'neighbors' of size 'count'
    # Assume 'count' is a power of 2
    for k in range(2, count + 1, 2):
        for j in range(k // 2, 0, -1):
            for i in range(count):
                l = i ^ j
                if l > i:
                    if (i & k) == 0 and neighbors[i] > neighbors[l]:
                        neighbors[i], neighbors[l] = neighbors[l], neighbors[i]
                    if (i & k) != 0 and neighbors[i] < neighbors[l]:
                        neighbors[i], neighbors[l] = neighbors[l], neighbors[i]


# CUDA kernel
@cuda.jit
def apply_median_filter_cuda(input_channel, output, kernel_size):
    x, y = cuda.grid(2)
    edge = kernel_size // 2
    if x >= edge and y >= edge and x < input_channel.shape[1] - edge and y < input_channel.shape[0] - edge:
        # Statically allocated array for neighbors
        neighbors = cuda.local.array(MAX_KERNEL_SIZE, dtype=uint8)
        count = 0
        for dy in range(-edge, edge + 1):
            for dx in range(-edge, edge + 1):
                neighbors[count] = input_channel[y + dy, x + dx]
                count += 1
        # Use bitonic sort
        bitonic_sort(neighbors, count)
        # Insertion sort to find the median
        # for i in range(1, count):
        #     key = neighbors[i]
        #     j = i - 1
        #     while j >= 0 and key < neighbors[j]:
        #         neighbors[j + 1] = neighbors[j]
        #         j -= 1
        #     neighbors[j + 1] = key
        # Assign median value
        output[y, x] = neighbors[count // 2]

def apply_median_filter(input_channel, kernel_size):
    # Convert input to device array
    input_channel_device = cuda.to_device(input_channel)
    output_device = cuda.device_array(input_channel.shape, dtype=np.uint8)

    # Define grid and block dimensions
    threadsperblock = (16, 16)
    # Calculate grid size to cover the whole image
    blockspergrid_x = int(np.ceil(input_channel.shape[1] / threadsperblock[1]))
    blockspergrid_y = int(np.ceil(input_channel.shape[0] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch kernel
    apply_median_filter_cuda[blockspergrid, threadsperblock](input_channel_device, output_device, kernel_size)
    
    # Copy the result back to host
    return output_device.copy_to_host()

def main():
    parser = argparse.ArgumentParser(description='Apply median filter to an image.')
    parser.add_argument('image_file', help='Path to the image file')

    args = parser.parse_args()

    # Read the image using the provided file path
    image_file_name = args.image_file
    image = cv2.imread(image_file_name, cv2.IMREAD_COLOR)
    # Read the image
    # image = cv2.imread('../resources/sp_img_gray_noise_heavy.png', cv2.IMREAD_COLOR)

    # Split the image into its color channels
    channels = cv2.split(image)

    # # Ensure that the CUDA context is initialized in the main thread before any multi-threaded operations.
    # dummy = cuda.device_array(1, dtype=np.uint8)

    start_time = time.time()
    
    # Apply median filter to each channel
    filtered_channels = [apply_median_filter(ch, 5) for ch in channels]  # Kernel size is 5

    end_time = time.time()
    print(f"Filtering time: {end_time - start_time} seconds")

    # Merge the channels back
    filtered_image = cv2.merge(filtered_channels)

    # Save the images before and after filtering
    cv2.imwrite('before_cuda_opt.jpg', image)
    cv2.imwrite('after_cuda_opt.jpg', filtered_image)

if __name__ == "__main__":
    main()
