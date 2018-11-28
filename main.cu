#include "lodepng.h"
#include <iostream>
#include <stdio.h>
#include <chrono>

__device__ 
void sort(unsigned char* input){
	for(int i = 0; i < 8; i++){
		int iMin = i;

		for(int j = i+1; j < 9; j++){
			if(input[j] < input[iMin]){
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
        for (int ox = -fsize; ox < fsize+1; ++ox) {
            for (int oy = -fsize; oy < fsize+1; ++oy) {
                if ((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
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

    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int x = offset % width;
    int y = (offset - x) / width;

    if (offset < width * height){

        unsigned char filterVectorRed[9] = {0,0,0,0,0,0,0,0,0};
        unsigned char filterVectorGreen[9] = {0,0,0,0,0,0,0,0,0};
        unsigned char filterVectorBlue[9] = {0,0,0,0,0,0,0,0,0};

        if (y == 0 || y == height - 1 || x == 0 || x == width - 1){
            output_image[offset * 3] = input_image[offset * 3];
            output_image[offset * 3 + 1] = input_image[offset * 3 + 1];
            output_image[offset * 3 + 2] = input_image[offset * 3 + 2];
        }
        else {
            int i = 0;
            for(int dx = -1; dx <= 1; dx++){
                for(int dy = -1; dy <= 1; dy++){
                    if(x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height){
                        const int currentOffset = (offset + dx + dy * width) * 3;
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

            output_image[offset * 3] = filterVectorRed[4];
            output_image[offset * 3 + 1] = filterVectorGreen[4];
            output_image[offset * 3 + 2] = filterVectorBlue[4];
        }
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

    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    switch (arg[0]) {
        /* Blur */
        case 'b':
        case 'B':
            start = std::chrono::high_resolution_clock::now();
            blur<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << duration.count() << " microseconds" << std::endl;  
            break;
            
        /* Greyscale */
        case 'g':
        case 'G':
            start = std::chrono::high_resolution_clock::now();
            greyscale<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << duration.count() << " microseconds" << std::endl; 
            break;
            
        /* Invert */
        case 'i':
        case 'I':
            start = std::chrono::high_resolution_clock::now();
            invert<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << duration.count() << " microseconds" << std::endl; 
            break;
        
        /* Median */
        case 'm':
        case 'M':
            start = std::chrono::high_resolution_clock::now();
            medianFilter<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << duration.count() << " microseconds" << std::endl; 
            break;
        
        /* Invalid Argument */
        default:
            printf("Invalid Argument. Options are: b, g, i, m\n");
            exit(1);
    }
    
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

    // Load the image
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    // Remove alpha channel
    unsigned char* input_image = new unsigned char[(in_image.size()*3)/4];
    unsigned char* output_image = new unsigned char[(in_image.size()*3)/4];
    int index = 0;
    for(int i = 0; i < in_image.size(); ++i) {
       if((i+1) % 4 != 0) {
           input_image[index] = in_image.at(i);
           output_image[index] = 255;
           index++;
       }
    }

    // Run the filter
    if (argc < 4) {
        printf("Invalid Usage\n");
        printf("Command should be of the form: ./filter input_image.png output_image.png <b, g, i, m>\n");
        exit(1);
    }
    else { filter(input_image, output_image, width, height, argv[3]); }

    // Reinsert alpha channel
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
