# Image Processing Using C++ and CUDA
The goal of this project is to apply filters or edits to existing images using CUDA to highly parallelize the compute operations.

## Timings Go Here

## Input Image
![Bevo](/images/bevo.png)

## Usage
This project requires:
* A [CUDA enabled NVIDIA GPU](https://developer.nvidia.com/cuda-gpus)
* The [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  * This toolkit includes `NVCC` - the NVIDIA C/C++ CUDA Compiler

To compile and run the project, execute the following commands:
```
cd /path/to/361C_Image_Processing
nvcc lodepng.cpp main.cu -o filter
./filter input_image.png output_image.png
```
where `input_image.png` is the original image and `output_image.png` is an existing png file to be replaced.

## Outputs 
### 1. Blur Image
![Invert](/images/blurbevo.png)
### 2. Invert Colors
![Invert](/images/evilbevo.png)
