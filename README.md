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
python meidan_filter_vedio_cuda.py python meidan_filter_vedio.py ../resources/video/videos_1_1080p.mp4
```

## Results

to profile nsys

```bash
nsys profile --stats=true ./cuda_median
```

### sp_img_gray_noise_heavy

| Method     | Total Time | Kernal Time | Transfer Time |
| ---------- | ---------- | ----------- | ------------- |
| OpenCV     |            |             |               |
| Seq C      |            |             |               |
| Seq Python | 2.8 sec    |             |               |
| CUDA C     |            |             |               |
| OpenACC    |            |             |               |
| CUDA Pythn | 0.37 sec   |             |               |
