#include "../ParseArgs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>

/**
* Allocates curand states.
*/
extern curandState* allocateCurandStates(int length);

/**
* Frees curandStates.
*/
extern cudaError_t freeCurandStates(curandState* dev_states);

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
* Alloctes a Float array to the cuda device.
*/
extern float* allocateAndCopyFloats(float* array,int length);

/**
* Frees Floats copied into a cuda array.
*/
extern cudaError_t freeFloats(float* dev_float);

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
* Responsible for calculating the dimenions of the gpu layout being used.
* @param numWords
* @param numHash
* @param device
*/
extern dim3 calculateThreadDimensions(int numWords,int numHash,int device);

/**
* Responsible for calculating the thread dimensions of the gpu layout.
* @param threadDimensions the dimensions of the thread block.
* @param numWords The total number of words being inserted.
* @param device The id of the device being used.
*/
extern dim3 calculateBlockDimensions(dim3 threadDimensions,int numWords, 
	int device);

/**
* Responsible for inserting words into the bloom filter.
*/
extern cudaError_t insertWords(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device);

/**
* Responsible for querying words inserted into the bloom filter
*/
extern cudaError_t queryWords(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device,
		char* result);

/**
* Responsible for calculating the dimenions of the gpu layout being used.
* @param numWords
* @param numHash
* @param device
*/
extern dim3 calculateThreadDimensions(int numWords,int numHash,int device);

/**
* Responsible for calculating the thread dimensions of the gpu layout.
* @param threadDimensions the dimensions of the thread block.
* @param device The id of the device being used.
*/
extern dim3 calculateBlockDimensions(dim3 threadDimensions,int numWords,
	int device);

/**
* Responsible for inserting words into the PBF.
*/
extern cudaError_t insertWordsPBF(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device,float prob);

