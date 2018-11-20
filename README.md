# Image Processing Using C++ and CUDA
The goal of this project is to apply filters or edits to existing images using CUDA to highly parallelize the compute operations.

## Timings Go Here

## Input Image
![Bevo](/images/bevo.png)

## Usage
This projects requires an NVIDIA GPU and a working installation of NVCC - the NVIDIA C/C++ CUDA Compiler.

To compile the project, run the following command in the main folder directory:
```
nvcc lodepng.cpp main.cu -o filter
```
## Outputs 
### 1. Blur Image
![Invert](/images/blurbevo.png)
### 2. Invert Colors
![Invert](/images/evilbevo.png)
