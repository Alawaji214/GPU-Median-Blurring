import cv2
import numpy as np
from numba import cuda
import time
from median_filter_cuda import apply_median_filter
import argparse

KERNAL_SIZE = 5
KERNAL_TOTAL_SIZE = KERNAL_SIZE * KERNAL_SIZE

def median_filter(frame):
    # Split the image into its color channels
    channels = cv2.split(frame)

    start_time = time.time()

    # Apply median filter to each channel
    filtered_channels = [apply_median_filter(ch, 5) for ch in channels]  # Kernel size is 5

    end_time = time.time()
    print(f"Filtering time: {end_time - start_time} seconds")
    filtered_image = cv2.merge(filtered_channels)
    return filtered_image
    # Implement median filter here, potentially using Numba for GPU acceleration
    # This is a placeholder as the actual GPU-accelerated implementation is non-trivial
    return cv2.medianBlur(frame, KERNAL_SIZE)

def main():
    parser = argparse.ArgumentParser(description='Apply median filter to an vedio.')
    parser.add_argument('vedio_file', help='Path to the vedio file')

    args = parser.parse_args()

    # Read the vedio using the provided file path
    vedio_file_name = args.vedio_file
    try:
        print("Start: Program")

        # Open video file
        capture = cv2.VideoCapture(vedio_file_name)
        if not capture.isOpened():
            print("Error opening video file")
            return -1

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup VideoWriter
        output = cv2.VideoWriter("output_" + vedio_file_name[vedio_file_name.rfind('/')+1:], cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width, frame_height))

        print("Start: Video")
        start_total = time.time()

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            # Apply median filter (need to implement GPU version if required)
            filtered_frame = median_filter(frame)

            # Write the frame into the output video
            output.write(filtered_frame)

        end_total = time.time()
        print("Finish: Video")

        # Release resources
        capture.release()
        output.release()

        total_duration = (end_total - start_total) * 1e6  # Convert to microseconds
        print(f"total time: {total_duration} us")

        print("Finished: Program")

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
