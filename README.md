# GPU Median Blurring

## Building

### Requirements

- OpenCV library
- Python and pip
- Nvidia GPU ( this readme assumses Nvidia L4)

#### Additional Requirements

For building and running CUDA and OpenACC versions, ensure the following are installed:

- [CUDA Toolkit (version 11.8 or above)](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk)

**Note**: Please make sure to add all dependencies to your PATH.

#### OpenCV Installation

install opencv library

```bash
sudo apt install libopencv-dev
```

#### Pything Libraries

for python, you have to install opencv through pip

```bash
pip3 install opencv-python
pip3 install numba

```

### C

#### OpenACC

To compile and run the OpenACC version:

Navigate to the OpenACC directory:

```bash
cd c/openacc
```

compile the application

```bash
nvc++ -acc -gpu=cc89 -Minfo=acc -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui median_filter.cpp -o median_filter_gpu
./median_filter_gpu
```

Or, for passing the image name as an argument:

```bash
nvc++ -acc -gpu=cc89 -Minfo=acc -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui median_filter_args.cpp -o median_filter_gpu_args
./median_filter_gpu_args <imageName>
```

To profile the OpenACC application:

```bash
Copy code
nsys profile -t openacc -b dwarf --stats=true ./median_filter
```

#### Cuda

goto c directary and use cmake to build the application

```bash
cd c
mkdir build && cd build
cmake ..
make
```

and then if everything run smothly, you can launch the application

```bash
./median #seq c
./image_cuda_median #cuda c
./video_cuda_median #cuda c
./video_cuda_median videos_2_1080p.mp4
```

or alternatively, you can use the following link to dirtectly build it

```bash
g++ -std=c++11 -I/usr/local/include/opencv4  -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui median_filter.cpp -o median_filter
```

## Python

nosiy image:
noisy_image1.jpg

```
cd python
pwd
/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true --output /home/bandr1994/COE506-Project/python/median_filter python median_filter.py ../resources/noisy_image1.jpg
/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true --output /home/bandr1994/COE506-Project/python/median_filter python median_filter_jit.py ../resources/noisy_image1.jpg
/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true python median_filter_multi.py ../resources/noisy_image1.jpg
/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true python median_filter_opencv.py ../resources/noisy_image1.jpg
/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true --output /home/bandr1994/COE506-Project/python/median_filter_cuda python median_filter_cuda.py ../resources/noisy_image1.jpg
/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true python median_filter_cuda_multi.py ../resources/noisy_image1.jpg
/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true python median_filter_cuda_opt.py ../resources/noisy_image1.jpg


/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true python meidan_filter_vedio.py ../resources/video/videos_1_1080p.mp4
/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true python meidan_filter_vedio_cuda.py ../resources/video/videos_1_1080p.mp4
```

```
/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true --output /home/bandr1994/COE506-Project/python/median_filter_cuda python median_filter_cuda.py ../resources/noisy_image2.jpg

/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true --output /home/bandr1994/COE506-Project/python/median_filter_cuda python median_filter_cuda.py ../resources/noisy_image3.jpg

/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true --output /home/bandr1994/COE506-Project/python/median_filter_cuda python median_filter_cuda.py ../resources/noisy_image4.jpg

/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/compilers/bin/nsys profile --stats=true --output /home/bandr1994/COE506-Project/python/median_filter_cuda python median_filter_cuda.py ../resources/noisy_image5.jpg
```

## Results

### Used GPU

NVIDIA L4

output of nvidia-smi

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA L4           On   | 00000000:00:03.0 Off |                    0 |
| N/A   47C    P8    17W /  72W |      0MiB / 23034MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                   
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

output of /usr/local/cuda-11.8/extras/demo_suite/deviceQuery

```bash
/usr/local/cuda-11.8/extras/demo_suite/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA L4"
  CUDA Driver Version / Runtime Version          12.0 / 11.8
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 22518 MBytes (23612293120 bytes)
MapSMtoCores for SM 8.9 is undefined.  Default to use 128 Cores/SM
MapSMtoCores for SM 8.9 is undefined.  Default to use 128 Cores/SM
  (58) Multiprocessors, (128) CUDA Cores/MP:     7424 CUDA Cores
  GPU Max Clock rate:                            2040 MHz (2.04 GHz)
  Memory Clock rate:                             6251 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 50331648 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 3
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.0, CUDA Runtime Version = 11.8, NumDevs = 1, Device0 = NVIDIA L4
Result = PASS

```

to profile nsys

```bash
nsys profile --stats=true ./cuda_median
```

### noisy_image1

| Method         | Total Time | Kernal Time | GPU Transfer Time |
| -------------- | ---------- | ----------- | ----------------- |
| OpenCV         |            |             |                   |
| Seq C          | 153 sec    |             |                   |
| CUDA C         | 0.053 sec  | 0.046 sec   | 0.007 sec         |
| OpenACC        | 0.256 sec  | 0.231 sec   | 0.026 sec         |
| CUDA Pythn     | 0.086 sec  | 0.060 sec   | 0.026 sec         |

### noisy_image2

| Method         | Total Time  | Kernal Time | GPU Transfer Time |
| -------------- | ----------- | ----------- | ----------------- |
| OpenCV         |             |             |                   |
| Seq C          | 152 sec     |             |                   |
| CUDA C         | 0.06409 sec | 0.05772 sec | 0.00637 sec       |
| OpenACC        | 0.26040 sec | 0.23297 sec | 0.02743 sec       |
| CUDA Pythn     | 0.08495 sec | 0.05970 sec | 0.02525 sec       |

### noisy_image3

| Method         | Total Time  | Kernal Time  | GPU Transfer Time |
| -------------- | ----------- | ------------ | ----------------- |
| OpenCV         |             |              |                   |
| Seq C          | 250 sec     |              |                   |
| CUDA C         | 0.10428 sec | 0.09386 sec  | 0.01042 sec       |
| OpenACC        | 0.42392 sec | 0.37996 sec  | 0.04396 sec       |
| CUDA Pythn     | 0.13956 sec | 0.09770 sec  | 0.04186 sec       |

### noisy_image4

| Method         | Total Time  | Kernal Time  | GPU Transfer Time |
| -------------- | ----------- | ------------ | ----------------- |
| OpenCV         |             |              |                   |
| Seq C          | 0.344 sec   |              |                   |
| CUDA C         | 0.00019 sec | 0.00017 sec  | 0.00002 sec       |
| OpenACC        | 0.00067 sec | 0.00051 sec  | 0.00015 sec       |
| CUDA Pythn     | 0.00022 sec | 0.00020 sec  | 0.00002 sec       |

### noisy_image5

| Method         | Total Time  | Kernal Time | GPU Transfer Time |
| -------------- | ----------- | ----------- | ----------------- |
| OpenCV         |             |             |                   |
| Seq C          | 0.58 sec    |             |                   |
| CUDA C         | 0.00035 sec | 0.00032 sec | 0.00003 sec       |
| OpenACC        | 0.00111 sec | 0.00088 sec | 0.00023 sec       |
| CUDA Pythn     | 0.00028 sec | 0.00025 sec | 0.00003 sec       

### sp_img_gray_noise_heavy

| Method     | Total Time | Kernal Time | GPU Transfer Time |
| ---------- | ---------- | ----------- | ----------------- |
| OpenCV     |            |             |                   |
| Seq C      | 0.73 sec   |             |                   |
| CUDA C     | 0.118 sec  | 0.001 sec   | 0.116 sec         |
| OpenACC    | 1.367 ms   | 1.171 ms    | 0.042 ms          |
| CUDA Pythn | 0.429 ms   | 0.208 ms    | 0.021 ms          |

## License

This project is licensed under the terms of the custom license. See the [LICENSE](LICENSE.md) file for details.
