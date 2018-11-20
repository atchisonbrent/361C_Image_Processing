#include "lodepng.h"
// #include "helper_cuda.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <algorithm>
#include <stdio.h>
#include <time.h>

#define TILE_W  16
#define TILE_H  16
#define R        2
#define D       (R*2+1)
#define S    (D*D)
#define BLOCK_W (TILE_W+(2*R))
#define BLOCK_H (TILE_H+(2*R))

#define CHANNELS 3

__device__ 
void sort(unsigned char* input){
	for(int i = 0; i < 8; i++){
		int iMin = i;

		for(int j = i+1; j < 9; j++){
			if(input[i] < input[iMin]){
				iMin = j;
			}
		}
		
		if(iMin != i){
			unsigned char temp = input[i];
			input[i] = input[iMin];
			input[iMin] = temp;
		}
	}
}

__global__
void mirror(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    
//    int col = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
//    int row = 3 * (blockIdx.y * blockDim.y + threadIdx.y);
//
//    if ( row >= width || col >= height ) { return; }
//
//    int col_new = col;
//    int row_new = width - row;
//
//    int myId = row * height + col;
//    int myId_new = row_new * height + col_new;
//
//    output_image[myId_new] = input_image[myId];
    
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    
    /* Check if Offset is Within Bounds */
    if (offset < width * height) {
        
        const int currentoffset = offset * 3;
        
        /* Get Current Color Values */
        float output_red = input_image[currentoffset];
        float output_green = input_image[currentoffset + 1];
        float output_blue = input_image[currentoffset + 2];
        
        /* Assign Inverted Color Values */
        output_image[offset * 3] = output_red;
        output_image[offset * 3 + 1] = output_green;
        output_image[offset * 3 + 2] = output_blue;
    }
}

__global__
void invert(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    
    /* Check if Offset is Within Bounds */
    if (offset < width * height) {
        
        const int currentoffset = offset * 3;
        
        /* Get Current Color Values */
        float output_red = input_image[currentoffset];
        float output_green = input_image[currentoffset + 1];
        float output_blue = input_image[currentoffset + 2];
        
        /* Assign Inverted Color Values */
        output_image[offset * 3] = 255 - output_red;
        output_image[offset * 3 + 1] = 255 - output_green;
        output_image[offset * 3 + 2] = 255 - output_blue;
    }
}

__global__
void h_average(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    
    /* Check if Offset is Within Bounds */
    if (offset < width * height) {
        
        const int currentoffset = offset * 3;
        
        /* Get Current Color Values */

        float output_red, output_green, output_blue;

        if(offset > 0 && offset < width*height - 1) {
            float output_red = (input_image[currentoffset] + input_image[(offset-1)*3] + input_image[(offset+1)*3])/3;
            float output_green = (input_image[currentoffset + 1] + input_image[(offset-1)*3 + 1] + input_image[(offset+1)*3 + 1])/3;
            float output_blue = (input_image[currentoffset + 2] + input_image[(offset-1)*3 + 2] + input_image[(offset+1)*3 + 2])/3;  
        }
        else {
            float output_red = input_image[currentoffset];
            float output_green = input_image[currentoffset + 1];
            float output_blue = input_image[currentoffset + 2];  
        }


        
        
        /* Assign Inverted Color Values */
        output_image[offset * 3] = 255 - output_red;
        output_image[offset * 3 + 1] = 255 - output_green;
        output_image[offset * 3 + 2] = 255 - output_blue;
    }
}

__global__
void greyscale(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    
    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    
    /* Check if Offset is Within Bounds */
    if (offset < width * height) {
        
        const int currentoffset = offset * 3;
        
        /* Get Current Color Values */
        float output_red = 0.21 * input_image[currentoffset];
        float output_green = 0.72 * input_image[currentoffset + 1];
        float output_blue = 0.07 * input_image[currentoffset + 2];
        float output_color = output_red + output_green + output_blue;
        
        /* Assign Inverted Color Values */
        output_image[offset * 3] = output_color;
        output_image[offset * 3 + 1] = output_color;
        output_image[offset * 3 + 2] = output_color;
    }
}

__global__ 
void simple_filter(int *input_image, int *g_odata, unsigned int width, unsigned int height){
    __shared__ int smem[BLOCK_W*BLOCK_H];
    int x = blockIdx.x*TILE_W + threadIdx.x - R;
    int y = blockIdx.y*TILE_H + threadIdx.y - R;

    x = max(0, x);
    x = min(x, width-1);
    y = max(y, 0);
    y = min(y, height-1);

    unsigned int index = y*width + x;
    unsigned int bindex = threadIdx.y*blockDim.y+threadIdx.x;

    smem[bindex] = input_image[index];
    __syncthreads();

    if((threadIdx.x >= R) && (threadIdx.x < (BLOCK_W-R)) && (threadIdx.y >= R) && (threadIdx.y < (BLOCK_H-R))){
        float sum = 0;
        for(int dy = -R; dy <= R; dy++){
            for(int dx = -R; dx <= R; dx++){
                float i = smem[bindex + (dy*blockDim.x) + dx];
                sum += i;
            }
        }
        g_odata[index] = sum/S;
    }
}

__global__
void blur(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset-x)/width;
    int fsize = 3; // Filter size
    if(offset < width*height) {

        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;
        int hits = 0;
        for(int ox = -fsize; ox < fsize+1; ++ox) {
            for(int oy = -fsize; oy < fsize+1; ++oy) {
                if((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                    const int currentoffset = (offset+ox+oy*width)*3;
                    output_red += input_image[currentoffset]; 
                    output_green += input_image[currentoffset+1];
                    output_blue += input_image[currentoffset+2];
                    hits++;
                }
            }
        }
        output_image[offset*3] = output_red/hits;
        output_image[offset*3+1] = output_green/hits;
        output_image[offset*3+2] = output_blue/hits;
        }
}

__global__ void
medianFilter(unsigned char* input_image, unsigned char* output_image, int width, int height){

	const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
	int x = offset % width;
	int y = (offset - x)/width;

	if(offset < width*height){
	
		unsigned char filterVectorRed[9] = {0,0,0,0,0,0,0,0,0};
		unsigned char filterVectorGreen[9] = {0,0,0,0,0,0,0,0,0};
		unsigned char filterVectorBlue[9] = {0,0,0,0,0,0,0,0,0};

		if(y == 0 || y == height - 1 || x == 0 || x == width - 1){
			output_image[offset*3] = input_image[offset];
			output_image[offset*3 + 1] = input_image[offset + 1];
			output_image[offset*3 + 2] = input_image[offset + 2];
		}
		else{
			int i = 0;
			for(int dx = -1; dx <= 1; dx++){
				for(int dy = -1; dy <= 1; dy++){
					if(x+dx >= 0 && x+dx < width && y+dy >= 0 && y+dy < height){
						const int currentOffset = (offset+dx+dy*width)*3;
						filterVectorRed[i] = input_image[currentOffset];
						filterVectorGreen[i] = input_image[currentOffset + 1];
						filterVectorBlue[i] = input_image[currentOffset + 2];
						i++;
					}
				}
			}
			sort(filterVectorRed);
			sort(filterVectorGreen);		
			sort(filterVectorBlue);

			output_image[offset*3] = filterVectorRed[4];
			output_image[offset*3 + 1] = filterVectorGreen[4];
			output_image[offset*3 + 2] = filterVectorBlue[4];
		}
	}

}

__device__ float exp(int i) { return exp((float) i); }

const int BLOCKDIM = 32;
const int sigma1 = 50;
const int sigma2 = 50;

__device__ const int FILTER_SIZE = 9;
__device__ const int FILTER_HALFSIZE = FILTER_SIZE >> 1;

__global__ 
void bilateral_filter_2d(unsigned char* input, unsigned char* output, int width, int height)
{
    const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset-x)/width;

	if(offset < width*height) {
		float running_total = 0;
		float norm_factor = 0;
		const int offset = y * width + x;
		for (int xctr = -FILTER_HALFSIZE; xctr <= FILTER_HALFSIZE; xctr++) 
		{
			for (int yctr = -FILTER_HALFSIZE; yctr <= FILTER_HALFSIZE; yctr++) 
			{
				int y_iter = y + xctr;
				int x_iter = x + yctr;
				if (x_iter < 0) x_iter = -x_iter;
				if (y_iter < 0) y_iter = -y_iter;
				if (x_iter > width-1) x_iter = width-1-xctr;
				if (y_iter > height-1) y_iter = height-1-yctr;
				float intensity_change = input[y_iter * width + x_iter] - input[y * width + x];
				float w1 = exp(-(xctr * xctr + yctr * yctr) / (2 * sigma1 * sigma1));
				float w2 = exp(-(intensity_change * intensity_change) / (2 * sigma2 * sigma2));
				running_total += input[y_iter * width + x_iter] * w1 * w2;
				norm_factor += w1 * w2;
			}
		}
        output[offset] = running_total / norm_factor;
        
	}
}

void getError(cudaError_t err) {
    if(err != cudaSuccess) {
        std::cout << "Error " << cudaGetErrorString(err) << std::endl;
    }
}

void filter (unsigned char* input_image, unsigned char* output_image, int width, int height, char* arg) {

    unsigned char* dev_input;
    unsigned char* dev_output;
    getError(cudaMalloc( (void**) &dev_input, width*height*3*sizeof(unsigned char)));
    getError(cudaMemcpy( dev_input, input_image, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice ));
 
    getError(cudaMalloc( (void**) &dev_output, width*height*3*sizeof(unsigned char)));

    /* Dimensions */
    dim3 blockDims(512, 1, 1);
    dim3 gridDims((unsigned int) ceil((double)(width*height * 3 / blockDims.x)), 1, 1 );

    // timet_t start, end;
    // start = clock();
    // end = clock();
    // std::cout << "Blur Filter took " << (end-start)/CLOCKS_PER_SEC << " ms\n";
    
    switch (arg[0]) {
        
        /* Blur */
        case 'b':
        case 'B':
            blur<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            
        /* Greyscale */
        case 'g':
        case 'G':
            greyscale<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            
        /* Invert */
        case 'i':
        case 'I':
            invert<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
        
        /* Median */
        case 'm':
        case 'M':
            medianFilter<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
        
        /* Invalid Argument */
        default:
            printf("Invalid Argument. Options are: b, g, i, m\n"); exit(1);
    
    getError(cudaMemcpy(output_image, dev_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost ));

    getError(cudaFree(dev_input));
    getError(cudaFree(dev_output));

}


int main(int argc, char *argv[]){
    // std::cout << argv[1];

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    std::vector<unsigned char> in_image;
    unsigned int width, height;

    // Load the data
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    // Prepare the data
    unsigned char* input_image = new unsigned char[(in_image.size()*3)/4];
    unsigned char* output_image = new unsigned char[(in_image.size()*3)/4];
    int where = 0;
    for(int i = 0; i < in_image.size(); ++i) {
       if((i+1) % 4 != 0) {
           input_image[where] = in_image.at(i);
           output_image[where] = 255;
           where++;
       }
    }

    // Run the filter on it
    if (argc < 4) {
        printf("Invalid Usage\n");
        printf("Command should be of the form: ./filter input_image.png output_image.png <b, g, i, m>\n");
        exit(1);
    }
    else { filter(input_image, output_image, width, height, argv[3]); }

    // Prepare data for output
    std::vector<unsigned char> out_image;
    for(int i = 0; i < in_image.size(); ++i) {
        out_image.push_back(output_image[i]);
        if((i+1) % 3 == 0) {
            out_image.push_back(255);
        }
    }
    
    // Output the data
    error = lodepng::encode(output_file, out_image, width, height);

    //if there's an error, display it
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    delete[] input_image;
    delete[] output_image;
    return 0;

}
