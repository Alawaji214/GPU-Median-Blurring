#include <opencv2/opencv.hpp>


#include <chrono>
#include "common.h"
#define KERNAL_SIZE 5
#define KERNAL_TOTAL_SIZE KERNAL_SIZE * KERNAL_SIZE

using namespace std::chrono;

__global__ void _median_filter(u_int8_t *channel, u_int8_t *out_channel, int size_x, int size_y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;       

    int edge = KERNAL_SIZE / 2;
    
    if( col > size_y || row > size_x) 
        return;
    
    if(row < edge || row > size_x - edge || col < edge || col > size_y - edge){
        // out_channel[row * size_y + col] = channel[row * size_y + col];
        return ;
    }
    
    // int avg
    u_int8_t pixelValues[KERNAL_TOTAL_SIZE];

    for (int y = 0; y < KERNAL_SIZE; y++) {
        for(int x = 0; x < KERNAL_SIZE; x++) {
            pixelValues[(y) * KERNAL_SIZE + (x)] = channel[(y+row-edge) * size_y + (x+col-edge)];
        }
    }


    // Get median pixel value and assign to filteredImage
    // std::nth_element(pixelValues.begin(), KERNAL_TOTAL_SIZE / 2, pixelValues.end());

    for (int i = 0; i < KERNAL_TOTAL_SIZE; i++) {
	    for (int j = i + 1; j < KERNAL_TOTAL_SIZE; j++) {
	        if (pixelValues[i] > pixelValues[j]) {
		        //Swap the variables.
		        u_int8_t tmp = pixelValues[i];
		        pixelValues[i] = pixelValues[j];
		        pixelValues[j] = tmp;
	        }
	    }
    }

    out_channel[row * size_y + col] = (u_int8_t) pixelValues[KERNAL_TOTAL_SIZE / 2];
}

void median_filter_driver(u_int8_t *channel, u_int8_t *out_channel, int size_x, int size_y) {

    // remove the edges
    dim3 threadsPerBlock(size_x, size_y);
    dim3 blocksPerGrid(1, 1);

    int maxthreadperblock = 1024; // TODO: take from cudaDeviceProp
    if(size_y * size_y > maxthreadperblock) {
        if(size_y > 32) {
            threadsPerBlock.y = 32;
            blocksPerGrid.y = ceil(double(size_y)/double(threadsPerBlock.y));           
        }
        threadsPerBlock.x = ceil(maxthreadperblock / threadsPerBlock.y);
        blocksPerGrid.x = ceil(double(size_y)/double(threadsPerBlock.x));
    }

    _median_filter<<<blocksPerGrid,threadsPerBlock>>>(channel,out_channel, size_x, size_y);
    CHECK_LAST_CUDA_ERROR();
}

int main() {

    try {
        std::cout << "Start: Program" << std::endl;

        cv::Mat image = cv::imread("resources/sp_img_gray_noise_heavy.png", cv::IMREAD_COLOR);

        int N_Channels = 3; // Number of Channels
        int rows = image.rows;
        int cols = image.cols;

        int gpu_device = -1;
        cudaGetDevice(&gpu_device);

        cv::Mat frame, filteredFrame;

        cv::Mat channels[N_Channels], outputChannels[N_Channels];

        // GPU device source and destination matrices
        u_int8_t *d_channels[N_Channels], *d_outputChannels[N_Channels];

        // // Save the frame before filtering
        std::string filenameBefore = "before.jpg" ;
        cv::imwrite(filenameBefore, image);

        cv::split(image, channels);

        std::cout << "Start: copying channels" << std::endl;

        auto start_cp = high_resolution_clock::now();

        for (int c = 0; c < N_Channels; c++) {
            cudaMallocManaged(&d_channels[c], sizeof(u_int8_t) * rows * cols);
            CHECK_LAST_CUDA_ERROR();
            cudaMallocManaged(&d_outputChannels[c], sizeof(u_int8_t) * rows * cols);
            CHECK_LAST_CUDA_ERROR();

            if(channels[c].isContinuous()) {
                cudaMemcpy(d_channels[c], channels[c].data, sizeof(u_int8_t) * rows * cols, cudaMemcpyHostToDevice);
                cudaMemPrefetchAsync(d_channels[c], sizeof(u_int8_t) * rows * cols, gpu_device);
                CHECK_LAST_CUDA_ERROR();
            }
            else {
                std::cout << "Error: Not Continuous" << std::endl;
            }
        }

        auto start_mf = high_resolution_clock::now();
        // Apply median filter to each channel
        for (int i = 0; i < N_Channels; i++) {
            median_filter_driver(d_channels[i], d_outputChannels[i], rows, cols);
            cudaMemPrefetchAsync(d_outputChannels[i], sizeof(u_int8_t) * rows * cols, cudaCpuDeviceId);
        }
        cudaDeviceSynchronize();
        auto end_mf = high_resolution_clock::now();

        for (int i = 0; i < N_Channels; i++) {
            outputChannels[i] = cv::Mat(rows, cols, CV_8UC1);
            cudaMemcpy(outputChannels[i].data, d_outputChannels[i], sizeof(u_int8_t) * rows*cols, cudaMemcpyDeviceToHost);
            CHECK_LAST_CUDA_ERROR();
        }

        auto end_cp = high_resolution_clock::now();

        // Merge the channels back
        cv::merge(outputChannels, N_Channels, filteredFrame);

        // Save the frame after filtering
        std::string filenameAfter = "after.jpg";
        cv::imwrite(filenameAfter, filteredFrame);
        
        std::cout << "End : Start Image Saving" << std::endl;

        auto total_duration = duration_cast<microseconds>(end_cp - start_cp).count();
        auto filter_duration = duration_cast<microseconds>(end_mf - start_mf).count();

        std::cout << "total time    : " << total_duration << " us" << std::endl;
        std::cout << "filter time   : " << filter_duration << " us" << std::endl;
        std::cout << "mem time      : " << total_duration - filter_duration << " us" << std::endl;
        for (int c = 0; c < N_Channels; c++) {
            cudaFree(d_channels[c]);
            CHECK_LAST_CUDA_ERROR();
            cudaFree(d_outputChannels[c]);
            CHECK_LAST_CUDA_ERROR();
        }
    } catch(const cv::Exception& ex) {
        std::cout << "Error: " << ex.what() << std::endl;
    }

    return 0;
}