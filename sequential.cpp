#include "lodepng.h"
#include <iostream>
#include <stdio.h>
#include <chrono>

void seqSort(unsigned char* input) {
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

void seqInvert(unsigned char* input_image, unsigned char* output_image, int width, int height){
    for(int i = 0; i < width*height; i++){
		int offset = i * 4;
        float output_red = input_image[offset];
        float output_green = input_image[offset + 1];
        float output_blue = input_image[offset + 2];

        output_image[offset] = 255 - output_red;
		output_image[offset + 1] = 255 - output_green;
		output_image[offset + 2] = 255 - output_blue;

    }
}

void seqGreyscale(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    for(int i = 0; i < width*height; i++){
		int offset = i * 4;
        float output_red = 0.21 * input_image[offset];
        float output_green = 0.72 * input_image[offset + 1];
        float output_blue = 0.07 * input_image[offset + 2];
		float output_color = output_red + output_green + output_blue;

        output_image[offset] = output_color;
		output_image[offset + 1] = output_color;
		output_image[offset + 2] = output_color;

    }
}

void seqBlur(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    int fsize = 3;  //filter size
    for(int i = 0; i < width*height; i++) {
        int x = i % width;
        int y = (i-x)/width;
        int hits = 0;
        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;
        for (int ox = -fsize; ox < fsize+1; ++ox){
            for (int oy = -fsize; oy < fsize+1; ++oy){
                if ((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height){
                    const int offset = (i+ox+oy*width)*4;
                    output_red += input_image[offset]; 
                    output_green += input_image[offset + 1];
                    output_blue += input_image[offset + 2];
                    hits++;
                }
            }
        }
        output_image[i*4] = output_red/hits;
        output_image[i*4 + 1] = output_green/hits;
        output_image[i*4 + 2] = output_blue/hits;
    }
}

void seqMedianFilter(unsigned char* input_image, unsigned char* output_image, int width, int height) {
    for(int i = 0; i < width*height; i++){
        int x = i % width;
        int y = (i - x) / width;

        unsigned char filterVectorRed[9] = {0,0,0,0,0,0,0,0,0};
        unsigned char filterVectorGreen[9] = {0,0,0,0,0,0,0,0,0};
        unsigned char filterVectorBlue[9] = {0,0,0,0,0,0,0,0,0};

        if (y == 0 || y == height - 1 || x == 0 || x == width - 1){
            output_image[i*4] = input_image[i];
            output_image[i*4 + 1] = input_image[i + 1];
            output_image[i*4 + 2] = input_image[i + 2];
        }
        else {
            int j = 0;
            for(int dx = -1; dx <= 1; dx++){
                for(int dy = -1; dy <= 1; dy++){
                    if(x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height){
                        const int currentOffset = (i + dx + dy * width) * 4;
                        filterVectorRed[j] = input_image[currentOffset];
                        filterVectorGreen[j] = input_image[currentOffset + 1];
                        filterVectorBlue[j] = input_image[currentOffset + 2];
                        j++;
                    }
                }
            }
            seqSort(filterVectorRed);
            seqSort(filterVectorGreen);
            seqSort(filterVectorBlue);

            output_image[i*4] = filterVectorRed[4];
            output_image[i*4 + 1] = filterVectorGreen[4];
            output_image[i*4 + 2] = filterVectorBlue[4];
        }
    }
}


void seqFilter (unsigned char* input_image, unsigned char* output_image, int width, int height, char* arg) {
	// auto start = std::chrono::high_resolution_clock::now();
	// seqMedianFilter(input_image, output_image, width, height);
	// auto stop = std::chrono::high_resolution_clock::now();
	// auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	// std::cout << duration.count() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    switch (arg[0]) {
        /* Blur */
        case 'b':
        case 'B':
            start = std::chrono::high_resolution_clock::now();
            seqBlur(input_image, output_image, width, height);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << duration.count() << " microseconds" << std::endl;  
            break;
            
        /* Greyscale */
        case 'g':
        case 'G':
            start = std::chrono::high_resolution_clock::now();
            seqGreyscale(input_image, output_image, width, height);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << duration.count() << " microseconds" << std::endl; 
            break;
            
        /* Invert */
        case 'i':
        case 'I':
            start = std::chrono::high_resolution_clock::now();
            seqInvert(input_image, output_image, width, height);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << duration.count() << " microseconds" << std::endl; 
            break;
        
        /* Median */
        case 'm':
        case 'M':
            start = std::chrono::high_resolution_clock::now();
            seqMedianFilter(input_image, output_image, width, height);
            stop = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << duration.count() << " microseconds" << std::endl;  
            break;
        
        /* Invalid Argument */
        default:
            printf("Invalid Argument. Options are: b, g, i, m\n");
            exit(1);
    }
}

int main(int argc, char *argv[]){
    const char* input_file = argv[1];
    const char* output_file = argv[2];

    std::vector<unsigned char> in_image;
    unsigned int width, height;

    // Load the image
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    // Prepare the image
    unsigned char* input_image = new unsigned char[in_image.size()*4];
    unsigned char* output_image = new unsigned char[in_image.size()*4];
    int where = 0;
    for(int i = 0; i < in_image.size(); i++) {
           input_image[i] = in_image.at(i);
           output_image[i] = in_image.at(i);
           where++;
    }

    // Run the filter
    if (argc < 4) {
        printf("Invalid Usage\n");
        printf("Command should be of the form: ./seq input_image.png output_image.png <b, g, i, m>\n");
        exit(1);
    }
    else{
        seqFilter(input_image, output_image, width, height, argv[3]);
    }
	
    // Prepare data for output to image
    std::vector<unsigned char> out_image;
    for(int i = 0; i < in_image.size(); i++) {
        out_image.push_back(output_image[i]);

    }
    
    // Output the data
    error = lodepng::encode(output_file, out_image, width, height);

    //if there's an error, display it
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    delete[] input_image;
    delete[] output_image;

    return 0;
}