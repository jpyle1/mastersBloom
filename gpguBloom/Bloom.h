#include "../ParseArgs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>


/**
* Allocates an Integer array to the cuda device.
* @param array The array of integers being allocated.
* @param length The number of items in the array.
*/
extern int* allocateAndCopyIntegers(int* array,int length);

/**
* Frees Integers copied into a cuda array.
* @param dev_array The integers copied into the cuda array. 
*/
extern cudaError_t freeIntegers(int* dev_array);

/**
* Allocates a character array to the cuda device.
* @param array
*/
extern char* allocateAndCopyChar(char* array,int length);

/**
* Fres Characters copied into a cuda array.
* @param dev_array The array being freed.
*/
extern cudaError_t freeChars(char* dev_array);

/**
* Copies a character array to the host.
* @param char* array The host array
* @param char8 dev_array The device array
* @param length The number of items being copied.
*/
extern cudaError_t copyCharsToHost(char* array,char* dev_array,int length);

/**
* Responsible for inserting words into the bloom filter.
*/
extern cudaError_t insertWords(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device);

/**
* Responsible for inserting words into the PBF bloom filter.
*/
extern cudaError_t insertWordsPBF(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device,float prob,
	int randOffset);


/**
* Responsible for querying words inserted into the bloom filter
*/
extern cudaError_t queryWords(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device,
		char* result);

/**
* Responsible for querying words inserted into the PBF bloom filter
*/
extern cudaError_t queryWordsPBF(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device,
		int* result);



