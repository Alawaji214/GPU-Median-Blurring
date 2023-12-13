#include <opencv2/opencv.hpp>

#include <chrono>
#include <string>
#include "common.h"
#include "median.h"
#define KERNAL_SIZE 5
#define KERNAL_TOTAL_SIZE KERNAL_SIZE * KERNAL_SIZE

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[]) {

    try {
        std::cout << "Start: Program" << std::endl;

        string image_name;
        if(argc > 1)
            image_name = argv[1];
        else
            image_name = "sp_img_gray_noise_heavy.png";

        cv::Mat image = cv::imread("resources/" + image_name, cv::IMREAD_COLOR);

        int N_Channels = 3; // Number of Channels
        int rows = image.rows;
        int cols = image.cols;

        int gpu_device = -1;
        cudaGetDevice(&gpu_device);

        cv::Mat frame, filteredFrame;

        cv::Mat channels[N_Channels], outputChannels[N_Channels];

        // GPU device source and destination matrices
        u_int8_t *h_channels[N_Channels], *h_outputChannels[N_Channels];
        u_int8_t *d_channels[N_Channels], *d_outputChannels[N_Channels];

        // // Save the frame before filtering
        std::string filenameBefore = "before.jpg" ;
        cv::imwrite(filenameBefore, image);

        cv::split(image, channels);

        std::cout << "Start: copying channels" << std::endl;

        auto start_cp = high_resolution_clock::now();

        for (int c = 0; c < N_Channels; c++) {
            cudaMallocHost(&h_channels[c], sizeof(u_int8_t) * rows * cols);
            CHECK_LAST_CUDA_ERROR();
            cudaMallocHost(&h_outputChannels[c], sizeof(u_int8_t) * rows * cols);
            CHECK_LAST_CUDA_ERROR();

            cudaMemcpy(h_channels[c], channels[c].data, sizeof(u_int8_t) * rows * cols, cudaMemcpyHostToHost);
        }
        std::cout << "Start: copying to device" << std::endl;
        
        for (int c = 0; c < N_Channels; c++) {
            cudaMalloc(&d_channels[c], sizeof(u_int8_t) * rows * cols);
            CHECK_LAST_CUDA_ERROR();
            cudaMalloc(&d_outputChannels[c], sizeof(u_int8_t) * rows * cols);
            CHECK_LAST_CUDA_ERROR();

            cudaMemcpyAsync(d_channels[c], h_channels[c], sizeof(u_int8_t) * rows * cols, cudaMemcpyHostToDevice);
            CHECK_LAST_CUDA_ERROR();
        }
        cudaDeviceSynchronize();

        std::cout << "Start: kernal" << std::endl;

        auto start_mf = high_resolution_clock::now();
        // Apply median filter to each channel
        for (int i = 0; i < N_Channels; i++) {
            median_filter_driver(d_channels[i], d_outputChannels[i], rows, cols);
        }
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        auto end_mf = high_resolution_clock::now();

        for (int i = 0; i < N_Channels; i++) {
            outputChannels[i] = cv::Mat(rows, cols, CV_8UC1);
            cudaMemcpyAsync(h_outputChannels[i], d_outputChannels[i], sizeof(u_int8_t) * rows*cols, cudaMemcpyDeviceToHost);
            CHECK_LAST_CUDA_ERROR();
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < N_Channels; i++) {
            cudaMemcpy(outputChannels[i].data, h_outputChannels[i], sizeof(u_int8_t) * rows*cols, cudaMemcpyHostToHost);
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
            cudaFreeHost(h_channels[c]);
            CHECK_LAST_CUDA_ERROR();
            cudaFreeHost(h_outputChannels[c]);
            CHECK_LAST_CUDA_ERROR();
        }
    } catch(const cv::Exception& ex) {
        std::cout << "Error: " << ex.what() << std::endl;
    }

    return 0;
}