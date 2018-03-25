# Direct Pose Estimation (DPE)
**Authors:** [Po-Chen Wu](http://media.ee.ntu.edu.tw/personal/pcwu/), [Hung-Yu Tseng](https://sites.google.com/site/hytseng0509/), [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/), and [Shao-Yi Chien](http://www.ee.ntu.edu.tw/profile?id=101)

Computer Vision and Image Understanding (CVIU), 2018

### Introduction
---
The DPE method is a planar object pose estimation algorithm using direct approach. It achieves state-of-the-art performance on three benchmark datasets. For more details and evaluation results, please check out our [project webpage](http://media.ee.ntu.edu.tw/research/DPE/) and [paper](http://media.ee.ntu.edu.tw/personal/pcwu/research/cviu2018_dpe/cviu2018_dpe.pdf).

![teaser](http://media.ee.ntu.edu.tw/research/DPE/images/teaser/total.jpg)

### Citation
---
If you find the code and datasets useful in your research, please cite:
    
    @article{DPE2018,
        author  = {Wu, Po-Chen and Tseng, Hung-Yu and Yang, Ming-Hsuan and Chien, Shao-Yi}, 
        title   = {Direct Pose Estimation for Planar Objects}, 
        journal = {Computer Vision and Image Understanding},
        year    = {2018}
    }

### Requirements and Dependencies
---
We have tested the program in **Ubuntu 16.04** and **Windows 10**, but it should be easy to compile in other platforms.

#### --- Basic (for CPU mode)
* MATLAB (we have tested with [**MATLAB R2017a**](https://www.mathworks.com/downloads/web_downloads/select_release?mode=gwylf))
* C++ compiler (we have tested with [**GCC 4.9.4**](https://packages.ubuntu.com/xenial/gcc-4.9) and [**Visual Studio 2015**](https://www.visualstudio.com/vs/older-downloads/)  C++compilers)

#### --- Optional (for GPU mode)
* CUDA (we have tested with [**CUDA 8.0**](https://developer.nvidia.com/cuda-toolkit-archive))
* Eigen  (we have tested with [**Eigen  3.3.4**](http://eigen.tuxfamily.org/index.php?title=Main_Page))
* OpenCV (we have tested with [**OpenCV 3.2.0**](https://opencv.org/releases.html))  

> Note that OpenCV should be compiled with **CUDA** support.  
> Be sure to enable **BUILD_opencv_world**.  
>> We recommend to configure an OpenCV project with **CMake**.  
>> ![cmake opencv](http://media.ee.ntu.edu.tw/research/DPE/images/cmake_opencv.png)

> *For **Windows** users, please build the **INSTALL** project especially.*


### Installation
---
Download repository:

    $ git clone https://github.com/pcwu0329/DPE.git

Run **DpeGui.m** in MATLAB. *It will automatically **download** associated **image files** if necessary.*

    # Start MATLAB
    $ matlab
    >> DpeGui

### Program Description
---
The proposed program can compute the planar target object pose in either **CPU** mode or **GPU** mode. *It will automatically **compile** associated **MEX files** if necessary.* Please take a look at the demo video below for more details.

[![demo video](http://img.youtube.com/vi/odlP_01DD4A/0.jpg)](http://www.youtube.com/watch?v=odlP_01DD4A)

For **Windows** users,  you may need to set paths to the **OpenCV install** folder and the **Eigen** folder in order to build the mex files successfully.  
![opencv install](http://media.ee.ntu.edu.tw/research/DPE/images/opencv_install_folder.png)
![opencv install](http://media.ee.ntu.edu.tw/research/DPE/images/eigen_folder.png)

For **Ubuntu** users, if you see the following error messages, you may have to preload the needed libraries.
```
Missing symbol 'th_comment_add' required by
'/usr/lib/x86_64-linux-gnu/libtheoraenc.so.1->/usr/lib/x86_64-linux-gnu/libavcodec-ffmpeg.so.56->/usr/local/lib/libopencv_world.so.3.2->/home/user/DPE/function/DPE_CUDA/apeCudaMex.mexa64'
Missing symbol 'th_comment_add_tag' required by
'/usr/lib/x86_64-linux-gnu/libtheoraenc.so.1->/usr/lib/x86_64-linux-gnu/libavcodec-ffmpeg.so.56->/usr/local/lib/libopencv_world.so.3.2->/home/user/DPE/function/DPE_CUDA/apeCudaMex.mexa64'
Missing symbol 'th_comment_clear' required by
'/usr/lib/x86_64-linux-gnu/libtheoraenc.so.1->/usr/lib/x86_64-linux-gnu/libavcodec-ffmpeg.so.56->/usr/local/lib/libopencv_world.so.3.2->/home/user/DPE/function/DPE_CUDA/apeCudaMex.mexa64'
Missing symbol 'th_comment_init' required by
'/usr/lib/x86_64-linux-gnu/libtheoraenc.so.1->/usr/lib/x86_64-linux-gnu/libavcodec-ffmpeg.so.56->/usr/local/lib/libopencv_world.so.3.2->/home/user/DPE/function/DPE_CUDA/apeCudaMex.mexa64'
Missing symbol 'th_comment_query' required by
'/usr/lib/x86_64-linux-gnu/libtheoraenc.so.1->/usr/lib/x86_64-linux-gnu/libavcodec-ffmpeg.so.56->/usr/local/lib/libopencv_world.so.3.2->/home/user/DPE/function/DPE_CUDA/apeCudaMex.mexa64'
Missing symbol 'th_comment_query_count' required by
'/usr/lib/x86_64-linux-gnu/libtheoraenc.so.1->/usr/lib/x86_64-linux-gnu/libavcodec-ffmpeg.so.56->/usr/local/lib/libopencv_world.so.3.2->/home/user/DPE/function/DPE_CUDA/apeCudaMex.mexa64'
Missing symbol '_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6resizeEmc' required by
'/usr/lib/x86_64-linux-gnu/libsnappy.so.1->/usr/lib/x86_64-linux-gnu/libavcodec-ffmpeg.so.56->/usr/local/lib/libopencv_world.so.3.2->/home/user/DPE/function/DPE_CUDA/apeCudaMex.mexa64'
Missing symbol '_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm' required by
'/usr/lib/x86_64-linux-gnu/libsnappy.so.1->/usr/lib/x86_64-linux-gnu/libavcodec-ffmpeg.so.56->/usr/local/lib/libopencv_world.so.3.2->/home/user/DPE/function/DPE_CUDA/apeCudaMex.mexa64'. 
``` 
To preload the needed libraries, terminate **MATLAB** and add the following content in **~/.bashrc**.
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libprotobuf.so.9:$LD_PRELOAD
```
Do not forget to execute `source ~/.bashrc` in the terminal to make the changes work immediately.

If it still does not work, then you may have to update your codecs.
```
sudo apt-get install ubuntu-restricted-extras
sudo apt-get install vlc
```