# COE506-Project

## Building

install opencv library

```bash
sudo apt install libopencv-dev
```

for python, you have to install opencv through pip

```bash
pip3 install opencv-python
pip3 install numba

```

### C

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

```
cd python
python median_filter.py ../resources/noise_intro_2.jpg
python median_filter_jit.py ../resources/noise_intro_2.jpg
python median_filter_multi.py ../resources/noise_intro_2.jpg
python median_filter_opencv.py ../resources/noise_intro_2.jpg
python median_filter_cuda.py ../resources/noise_intro_2.jpg
python median_filter_cuda_multi.py ../resources/noise_intro_2.jpg

python meidan_filter_vedio.py ../resources/video/videos_1_1080p.mp4
python meidan_filter_vedio_cuda.py ../resources/video/videos_1_1080p.mp4
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

| Method     | Total Time | Kernal Time | GPU Transfer Time |
| ---------- | ---------- | ----------- | ----------------- |
| OpenCV     |            |             |                   |
| Seq C      | 153 sec    |             |                   |
| Seq Python |            |             |                   |
| CUDA C     | 0.38 sec   | 0.098 sec   | 0.28 sec          |
| OpenACC    |            |             |                   |
| CUDA Pythn |            |             |                   |

### sp_img_gray_noise_heavy

| Method     | Total Time | Kernal Time | GPU Transfer Time |
| ---------- | ---------- | ----------- | ----------------- |
| OpenCV     |            |             |                   |
| Seq C      | 0.73 sec   |             |                   |
| Seq Python | 2.8 sec    |             |                   |
| CUDA C     | 0.118 sec  | 0.001 sec   | 0.116 sec         |
| OpenACC    |            |             |                   |
| CUDA Pythn | 0.37 sec   |             |                   |
