#include <opencv2/opencv.hpp>

#include <chrono>
#include <string>
#include "median.h"
#include "common.h"

#define KERNAL_SIZE 5
#define KERNAL_TOTAL_SIZE KERNAL_SIZE * KERNAL_SIZE


using namespace std::chrono;
using namespace std;

int main(int argc, char *argv[]) {
    try {
        std::cout << "Start: Program" << std::endl;
        
        string video_name;
        if(argc > 1)
            video_name = argv[1];
        else
            video_name = "videos_1_1080p.mp4";


        cv::VideoCapture capture("resources/video/" + video_name);
        if (!capture.isOpened()) {
            std::cerr << "Error opening video file" << std::endl;
            return -1;
        }

        int frame_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));    
        int frame_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

        cv::VideoWriter output("output_" + video_name, 
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 10, cv::Size(frame_width, frame_height));
        
        
        int N_Channels = 3; // Number of Channels
        cv::Mat frame, filteredFrame, channels[N_Channels], outputChannels[N_Channels];

        int gpu_device = -1;
        cudaGetDevice(&gpu_device);

        // GPU device source and destination matrices
        u_int8_t *d_channels[N_Channels], *d_outputChannels[N_Channels];

        for (int c = 0; c < N_Channels; c++) {
            cudaMallocManaged(&d_channels[c], sizeof(u_int8_t) * frame_height * frame_width);
            CHECK_LAST_CUDA_ERROR();
            cudaMallocManaged(&d_outputChannels[c], sizeof(u_int8_t) * frame_height * frame_width);
            CHECK_LAST_CUDA_ERROR();
            outputChannels[c] = cv::Mat(frame_height, frame_width, CV_8UC1);
        }

        // loop through the frames
        int f_ctr = 0;
        std::cout << "Start: Video" << std::endl;

        auto start_total = high_resolution_clock::now();

        while(true) {
            // Read a new frame from the video

            bool isSuccess = capture.read(frame);
            if (!isSuccess) {
                break;    
            }
            std::cout << "Read: Frame " << ++f_ctr << std::endl;

            // Merge the channels back
            cv::split(frame, channels);

            // copy the channels to the device
            for(int c = 0; c < N_Channels; c++) {
                if(channels[c].isContinuous()) {
                    cudaMemcpy(d_channels[c], channels[c].data, sizeof(u_int8_t) * frame_height * frame_width, cudaMemcpyHostToDevice);
                    cudaMemPrefetchAsync(d_channels[c], sizeof(u_int8_t) * frame_height * frame_width, gpu_device);
                    CHECK_LAST_CUDA_ERROR();
                }
                else {
                    std::cout << "Error: Not Continuous" << std::endl;
                }
            }

            // Apply median filter to each channel
            for (int c = 0; c < N_Channels; c++) {
                median_filter_driver(d_channels[c], d_outputChannels[c], frame_height, frame_width);
                cudaMemPrefetchAsync(d_outputChannels[c], sizeof(u_int8_t) * frame_height * frame_width, cudaCpuDeviceId);
            }
            cudaDeviceSynchronize();

            // copy back the result
            for (int c = 0; c < N_Channels; c++) {
                cudaMemcpy(outputChannels[c].data, d_outputChannels[c], sizeof(u_int8_t) * frame_height * frame_width, cudaMemcpyDeviceToHost);
                CHECK_LAST_CUDA_ERROR();
            }

            // Write the frame into the output video
            cv::merge(outputChannels, 3, filteredFrame);
            output.write(filteredFrame);
        }

        auto end_total = high_resolution_clock::now();
        std::cout << "Finish: Video" << std::endl;

        // Release the video capture and writer
        capture.release();
        output.release();


        auto total_duration = duration_cast<microseconds>(end_total - start_total).count();

        std::cout << "total time    : " << total_duration << " us" << std::endl;

        for (int c = 0; c < N_Channels; c++) {
            cudaFree(d_channels[c]);
            CHECK_LAST_CUDA_ERROR();
            cudaFree(d_outputChannels[c]);
            CHECK_LAST_CUDA_ERROR();
        }
        std::cout << "Finished: Program" << std::endl;

    } catch(const cv::Exception& ex) {
        std::cout << "Error: " << ex.what() << std::endl;
    }

    return 0;
}