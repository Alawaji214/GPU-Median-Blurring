#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>

uchar findMedian(uchar *arr, int n)
{
    // Simple insertion sort
    for (int i = 1; i < n; i++)
    {
        uchar key = arr[i];
        int j = i - 1;

        // Move elements of arr[0..i-1], that are greater than key,
        // to one position ahead of their current position
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }

    // Return the median
    if (n % 2 != 0) // If odd number of elements
        return arr[n / 2];
    else // If even number of elements
        return (arr[(n - 1) / 2] + arr[n / 2]) / 2;
}

// Function to apply median filter to a single channel
void applyMedianFilter(uchar *input, uchar *output, int rows, int cols, int kernelSize)
{
    int edge = kernelSize / 2;
    int windowSize = kernelSize * kernelSize;
    uchar *neighbors = new uchar[windowSize];

#pragma acc data copyin(input[0 : rows * cols]) copyout(output[0 : rows * cols])
    {
#pragma acc parallel loop
        for (int y = edge; y < rows - edge; y++)
        {
#pragma acc loop
            for (int x = edge; x < cols - edge; x++)
            {
                int count = 0;
                for (int dy = -edge; dy <= edge; dy++)
                {
                    for (int dx = -edge; dx <= edge; dx++)
                    {
                        int idx = (y + dy) * cols + (x + dx);
                        neighbors[count++] = input[idx];
                    }
                }

                uchar median = findMedian(neighbors, windowSize);
                output[y * cols + x] = median;
                // std::nth_element(neighbors, neighbors + windowSize / 2, neighbors + windowSize);
                // output[y * cols + x] = input[y * cols + x] ;//neighbors[windowSize / 2];
                // std::cout << "("<<x<<","<<y<<")"<< " : "<<output[y * cols + x] << std::endl;
            }
        }
    }

    delete[] neighbors;
}

void applyMedianFilter2(uchar *input, uchar *output, int rows, int cols, int kernelSize, uchar *neighbors)
{
    int edge = kernelSize / 2;
    int windowSize = kernelSize * kernelSize;

#pragma acc data copyin(input[0 : rows * cols]) copyout(output[0 : rows * cols])
    {
#pragma acc parallel loop private(neighbors)
        for (int y = edge; y < rows - edge; y++)
        {
#pragma acc loop
            for (int x = edge; x < cols - edge; x++)
            {
                int count = 0;
                for (int dy = -edge; dy <= edge; dy++)
                {
                    for (int dx = -edge; dx <= edge; dx++)
                    {
                        int idx = (y + dy) * cols + (x + dx);
                        neighbors[count++] = input[idx];
                    }
                }
                std::nth_element(neighbors, neighbors + windowSize / 2, neighbors + windowSize);
                output[y * cols + x] = neighbors[windowSize / 2];
            }
        }
    }
}

void applyMedianFilter3(uchar *input, uchar *output, int rows, int cols, int kernelSize)
{
    int edge = kernelSize / 2;
    int windowSize = kernelSize * kernelSize;

    // #pragma acc data copyin(input[0:rows*cols]) copyout(output[0:rows*cols])
    {
        // #pragma acc parallel loop
        for (int y = edge; y < rows - edge; y++)
        {
            // #pragma acc loop
            for (int x = edge; x < cols - edge; x++)
            {
                uchar neighbors[windowSize]; // Declare inside the loop
                int count = 0;
                for (int dy = -edge; dy <= edge; dy++)
                {
                    for (int dx = -edge; dx <= edge; dx++)
                    {
                        int idx = (y + dy) * cols + (x + dx);
                        neighbors[count++] = input[idx];
                    }
                }
                // Use an alternative to std::nth_element if necessary
                // output[y * cols + x] = neighbors[windowSize / 2]; // Simplified for demonstration
                output[y * cols + x] = output[y * cols + x];
            }
        }
    }
}

void applyMedianFilter4(uchar *input, uchar *output, int rows, int cols, int kernelSize)
{
    int edge = kernelSize / 2;
    int windowSize = kernelSize * kernelSize;
    uchar neighbors[windowSize];
    // #pragma acc data copyin(input[0:rows*cols]) copyout(output[0:rows*cols])
    {
#pragma acc parallel loop collapse(2) \
    copyin(input[0 : rows * cols]) \
    copyout(output[0 : rows * cols]) \
    private(neighbors) \
    num_gangs(132) /* 4 times the number of SMs to keep the GPU busy */ \
    num_workers(32) /* One worker per warp */ \
    vector_length(32) /* Equal to the warp size */
        for (int y = edge; y < rows - edge; y++)
        {
            for (int x = edge; x < cols - edge; x++)
            {
                int count = 0;
                for (int dy = -edge; dy <= edge; dy++)
                {
                    for (int dx = -edge; dx <= edge; dx++)
                    {
                        int idx = (y + dy) * cols + (x + dx);
                        neighbors[count++] = input[idx];
                    }
                }

                uchar median = findMedian(neighbors, windowSize);
                output[y * cols + x] = median;
                //printf("GPU (%d,%d) : %d\n", x, y, output[y * cols + x]);
            }
        }

        // #pragma acc wait
    }
    // #pragma acc wait
    // #pragma acc update host(output[0:rows*cols])

}

void applyMedianFilter5(uchar *input[3], uchar *output[3], int rows, int cols, int kernelSize)
{
    int edge = kernelSize / 2;
    int windowSize = kernelSize * kernelSize;


    {
#pragma acc parallel loop collapse(3) copyin(input[0 : 3][0 : rows * cols]) copyout(output[0 : 3][0 : rows * cols]) 
        for (int c = 0; c < 3; c++)
        {
            for (int y = edge; y < rows - edge; y++)
            {
                for (int x = edge; x < cols - edge; x++)
                {
                    uchar neighbors[windowSize];
                    int count = 0;
                    for (int dy = -edge; dy <= edge; dy++)
                    {
                        for (int dx = -edge; dx <= edge; dx++)
                        {
                            int idx = (y + dy) * cols + (x + dx);
                            neighbors[count++] = input[c][idx];
                        }
                    }
                    uchar median = findMedian(neighbors, windowSize);
                    output[c][y * cols + x] = median;
                }
            }
        }
    }
}

void applyMedianFilter6(uchar *input[3], uchar *output[3], int rows, int cols, int kernelSize)
{
    int edge = kernelSize / 2;
    int windowSize = kernelSize * kernelSize;
    uchar neighbors[3][windowSize];


    {
#pragma acc parallel loop collapse(3) copyin(input[0 : 3][0 : rows * cols]) copyout(output[0 : 3][0 : rows * cols]) private(neighbors)
        for (int c = 0; c < 3; c++)
        {
            for (int y = edge; y < rows - edge; y++)
            {
                for (int x = edge; x < cols - edge; x++)
                {
                    
                    int count = 0;
                    for (int dy = -edge; dy <= edge; dy++)
                    {
                        for (int dx = -edge; dx <= edge; dx++)
                        {
                            int idx = (y + dy) * cols + (x + dx);
                            neighbors[c][count++] = input[c][idx];
                        }
                    }
                    uchar median = findMedian(neighbors[c], windowSize);
                    output[c][y * cols + x] = median;
                }
            }
        }
    }
}

void applyMedianFilter7(uchar *input[3], uchar *output[3], int rows, int cols, int kernelSize) {
    int edge = kernelSize / 2;
    int windowSize = kernelSize * kernelSize;

    // Process each channel independently
    for (int c = 0; c < 3; c++) {
        uchar neighbors[windowSize]; // Separate neighbors array for each channel

        #pragma acc parallel loop copyin(input[c][0:rows*cols]) copyout(output[c][0:rows*cols]) private(neighbors)
        for (int y = edge; y < rows - edge; y++) {
            for (int x = edge; x < cols - edge; x++) {
                int count = 0;
                for (int dy = -edge; dy <= edge; dy++) {
                    for (int dx = -edge; dx <= edge; dx++) {
                        int idx = (y + dy) * cols + (x + dx);
                        neighbors[count++] = input[c][idx];
                    }
                }
                uchar median = findMedian(neighbors, windowSize);
                output[c][y * cols + x] = median;
            }
        }
    }
}



int main()
{
    cv::Mat image = cv::imread("resources/noisy_image1.jpg", cv::IMREAD_COLOR);

    cv::Mat frame, filteredFrame;
    cv::Mat channels[3], outputChannels[3];

    // Split the frame into its color channels
    cv::split(image, channels);

    double t0 = static_cast<double>(cv::getTickCount());

    // Apply median filter to each channel
    outputChannels[0] = cv::Mat(image.rows, image.cols, CV_8UC1);
    outputChannels[1] = cv::Mat(image.rows, image.cols, CV_8UC1);
    outputChannels[2] = cv::Mat(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < 3; i++) {
        applyMedianFilter4(channels[i].data, outputChannels[i].data, image.rows, image.cols, 5); // Kernel size is 5
    }

    /*
       uchar *inputChannels[3] = {channels[0].data, channels[1].data, channels[2].data};
    uchar *outputChannelsData[3] = {outputChannels[0].data, outputChannels[1].data, outputChannels[2].data};

    applyMedianFilter5(inputChannels, outputChannelsData, image.rows, image.cols, 5);
    */
 

    double t1 = static_cast<double>(cv::getTickCount());
    double elapsed = (t1 - t0) / cv::getTickFrequency();
    std::cout << "Time elapsed: " << elapsed << " seconds." << std::endl;

    // Merge the channels back
    cv::merge(outputChannels, 3, filteredFrame);

    // Save the frame after filtering
    std::string filenameAfter = "after.jpg";
    cv::imwrite(filenameAfter, filteredFrame);

    // Save the frame before filtering
    std::string filenameBefore = "before.jpg";
    cv::imwrite(filenameBefore, image);

    return 0;
}
