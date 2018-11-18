//
//  simple-filter.cpp
//  term_project
//
//  Created by Brent Atchison on 11/18/18.
//  Copyright © 2018 Brent Atchison. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma pack(push, 1)

typedef struct tagBITMAPFILEHEADER {
    int bfType;         //specifies the file type
    double bfSize;      //specifies the size in bytes of the bitmap file
    int bfReserved1;    //reserved; must be 0
    int bfReserved2;    //reserved; must be 0
    double bfOffBits;   //species the offset in bytes from the bitmapfileheader to the bitmap bits
} BITMAPFILEHEADER;

#pragma pack(pop)

#pragma pack(push, 1)

typedef struct tagBITMAPINFOHEADER {
    double biSize;  //specifies the number of bytes required by the struct
    long biWidth;  //specifies width in pixels
    long biHeight;  //species height in pixels
    int biPlanes; //specifies the number of color planes, must be 1
    int biBitCount; //specifies the number of bit per pixel
    double biCompression;//spcifies the type of compression
    double biSizeImage;  //size of image in bytes
    long biXPelsPerMeter;  //number of pixels per meter in x axis
    long biYPelsPerMeter;  //number of pixels per meter in y axis
    double biClrUsed;  //number of colors used by th ebitmap
    double biClrImportant;  //number of colors that are important
} BITMAPINFOHEADER;

#pragma pack(pop)

unsigned char *LoadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader)
{
    FILE *filePtr; //our file pointer
    BITMAPFILEHEADER bitmapFileHeader; //our bitmap file header
    unsigned char *bitmapImage;  //store image data
    int imageIdx=0;  //image index counter
    unsigned char tempRGB;  //our swap variable
    
    //open filename in read binary mode
    filePtr = fopen(filename,"rb");
    if (filePtr == NULL)
        return NULL;
    
    //read the bitmap file header
    fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER),1,filePtr);
    
    //verify that this is a bmp file by check bitmap id
    if (bitmapFileHeader.bfType !=0x4D42)
    {
        fclose(filePtr);
        return NULL;
    }
    
    //read the bitmap info header
    fread(bitmapInfoHeader, sizeof(BITMAPINFOHEADER),1,filePtr); // small edit. forgot to add the closing bracket at sizeof
    
    //move file point to the begging of bitmap data
    fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);
    
    //allocate enough memory for the bitmap image data
    bitmapImage = (unsigned char*)malloc(bitmapInfoHeader->biSizeImage);
    
    //verify memory allocation
    if (!bitmapImage)
    {
        free(bitmapImage);
        fclose(filePtr);
        return NULL;
    }
    
    //read in the bitmap image data
    fread(bitmapImage,bitmapInfoHeader->biSizeImage,filePtr);
    
    //make sure bitmap image data was read
    if (bitmapImage == NULL)
    {
        fclose(filePtr);
        return NULL;
    }
    
    //swap the r and b values to get RGB (bitmap is BGR)
    for (imageIdx = 0;imageIdx < bitmapInfoHeader->biSizeImage;imageIdx+=3) // fixed semicolon
    {
        tempRGB = bitmapImage[imageIdx];
        bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
        bitmapImage[imageIdx + 2] = tempRGB;
    }
    
    //close file and return bitmap iamge data
    fclose(filePtr);
    return bitmapImage;
}

BITMAPINFOHEADER bitmapInfoHeader;
unsigned char *bitmapData;
