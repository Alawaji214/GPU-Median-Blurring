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