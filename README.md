# Image Processing Using C++ and CUDA
The goal of this project is to apply filters or edits to existing images using CUDA to highly parallelize the compute operations.

---

## Input Image
![Bevo](/images/bevo.png)

---

## Usage
1. This project requires:
* A [CUDA enabled NVIDIA GPU](https://developer.nvidia.com/cuda-gpus)
* The [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  * This toolkit includes `NVCC` - the NVIDIA C/C++ CUDA Compiler
---
2. To **compile** the project, execute the following commands:
```
git clone https://github.com/atchisonbrent/361C_Image_Processing.git\

cd /path/to/361C_Image_Processing

nvcc lodepng.cpp main.cu -o filter
```
---
3. To **execute** the project, run the following command:

For **Windows**:
```
filter.exe input_image.png output_image.png <b, g, i>
```
For **Mac** or **Linux**:
```
./filter input_image.png output_image.png <b, g, i>
```
Where `input_image.png` is the original image and `output_image.png` is an existing png file to be replaced.

`<b, g, i>` correspond to `blur`, `greyscale`, and `invert` respectively.

---

## Outputs 
### 1. Blur Image
![Blur](/images/blurbevo.png)
### 2. Invert Colors
![Invert](/images/evilbevo.png)
### 3. Greyscale
![Greyscale](/images/greybevo.png)
