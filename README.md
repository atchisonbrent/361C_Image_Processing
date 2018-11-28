# Image Processing Using C++ and CUDA
The goal of this project is to apply filters or edits to existing images using CUDA to highly parallelize the compute operations.

***

## Input Image
![Bevo](/images/bevo.png)

***

## Usage
1. This project requires:
* A [CUDA-Enabled NVIDIA GPU](https://developer.nvidia.com/cuda-gpus)
* The [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  * This toolkit includes `NVCC` - the NVIDIA C/C++ CUDA Compiler
---
2. To **compile** the project, execute the following commands:
```
git clone https://github.com/atchisonbrent/361C_Image_Processing.git

cd /path/to/361C_Image_Processing

nvcc lodepng.cpp main.cu -o filter
```
---
3. To **execute** the project, run the following command:

For **Windows**:
```
filter.exe input_image.png output_image.png <b, g, i, m>
```
For **Mac** or **Linux**:
```
./filter input_image.png output_image.png <b, g, i, m>
```
Where `input_image.png` is the original image and `output_image.png` is an existing png file to be replaced.

`<b, g, i, m>` correspond to `blur`, `greyscale`, `invert`, and `median` respectively.

***

## Outputs 
### 1. Blur Image
![Blur](/images/blurbevo.png)
### 2. Invert Colors
![Invert](/images/evilbevo.png)
### 3. Greyscale
![Greyscale](/images/greybevo.png)
### 4. Median Restore
#### Before
![Input](/images/med_before.png)
#### After
![Output](/images/med_after.png)

***

## Timings
### 1. Blur Image
#### CUDA
Average of 10 Iterations: 35.3 microseconds
#### Sequential
Average of 10 Iterations: 102069.4 microseconds
### 2. Invert Colors
#### CUDA
Average of 10 Iterations: 38.4 microseconds
#### Sequential
Average of 10 Iterations:  3498.8 microseconds
### 3. Greyscale
#### CUDA
Average of 10 Iterations: 43 microseconds
#### Sequential
Average of 10 Iterations: 3998 microseconds
### 4. Median Restore
#### CUDA
Average of 10 Iterations: 29 microseconds
#### Sequential
Average of 10 Iterations: 29155.1 microseconds