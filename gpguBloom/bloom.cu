#include "Bloom.h"

/**
* Allocates an Integer array to the cuda device.
* @param array The array of integers being allocated.
* @param length The number of items in the array.
*/
int* allocateAndCopyIntegers(int* array,int length){
	int* dev_int;
	cudaError_t result = cudaMalloc((void**)&dev_int,length*sizeof(int));
	if(result!=cudaSuccess){
		return 0;
	}	 
	result = cudaMemcpy(dev_int,array,sizeof(int)*length,
		cudaMemcpyHostToDevice);
	if(result!=cudaSuccess){
		return 0;
	}	
	return dev_int; 
}

/**
* Frees Integers copied into a cuda array.
* @param dev_array The integers copied into the cuda array. 
*/
cudaError_t freeIntegers(int* dev_array){
	return cudaFree(dev_array);
}

/**
* Allocates a character array to the cuda device.
* @param array
*/
char* allocateAndCopyChar(char* array,int length){
	char* dev_array;
	cudaError_t result = cudaMalloc((void**)&dev_array,length*sizeof(char));
	if(result!=cudaSuccess){
		return 0;
	}	 
	result = cudaMemcpy(dev_array,array,sizeof(char)*length,
		cudaMemcpyHostToDevice);
	if(result!=cudaSuccess){
		return 0;
	}	
	return dev_array; 
}

/**
* Fres Characters copied into a cuda array.
* @param dev_array The array being freed.
*/
cudaError_t freeChars(char* dev_array){
	return cudaFree(dev_array);
}

/**
* Copies a character array to the host.
* @param char* array The host array
* @param char8 dev_array The device array
* @param length The number of items being copied.
*/
cudaError_t copyCharsToHost(char* array,char* dev_array,int length){
	return cudaMemcpy(array,dev_array,sizeof(char)*length,cudaMemcpyDeviceToHost);
}

/**
* Responsible for calculating the dimenions of the gpu layout being used.
* @param numWords
* @param numHash
* @param device
*/
dim3 calculateThreadDimensions(int numWords,int numHash,int device){
	if(numWords == 0 || numHash == 0){
		printf("Nothing to do \n");
		return dim3(0,0,0);
	}
	//Get the properties of the device the user selected.
	cudaDeviceProp deviceProps;
  cudaGetDeviceProperties(&deviceProps, device);
		
	//Firstly, solve for the max number of words that 
	//Can be processed in one thread block.
	int maxWordPerBlock = deviceProps.maxThreadsPerBlock/numHash;
	
	//Check to see if the user specified too many hash functions.
	if(maxWordPerBlock ==0){
		printf("Too many hash functions \n");
		return dim3(0,0); 
	}
	int wordsPerBlock = 32*(maxWordPerBlock/32);
	if(wordsPerBlock ==0)
		wordsPerBlock = maxWordPerBlock;
	dim3 threadDimensions(wordsPerBlock,numHash);
	return threadDimensions;
}

/**
* Responsible for calculating the thread dimensions of the gpu layout.
* @param threadDimensions the dimensions of the thread block.
* @param device The id of the device being used.
*/
dim3 calculateBlockDimensions(dim3 threadDimensions,int numWords,
	int device){
	if(numWords == 0){
		printf("Nothing to do \n");
		return dim3(0,0,0);
	}

	//Get the device information being used.
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps,device);
	//Calculate the number of blocks needed to process all of the words.
	int numBlocksNeeded = numWords/threadDimensions.x;
	if(numWords%threadDimensions.x!=0)
		numBlocksNeeded++;
	//Hard coded due to hydra glitch.
	int maxGridSizeX = 65535;
	int numBlocksPerRow;	
	
	if(numBlocksNeeded<=maxGridSizeX)
		numBlocksPerRow = numBlocksNeeded;
	else{
		numBlocksPerRow = maxGridSizeX;
	}

	int numRows = numBlocksNeeded/numBlocksPerRow;
	if(numBlocksNeeded%numBlocksPerRow!=0){
		numRows++;
	}
	if(numRows>deviceProps.maxGridSize[1]){
		printf("Too many rows requested %i, \n",numRows);
		printf("Blocks Per Row %i \n",numBlocksPerRow);
		printf("threadDim: %i,%i \n",threadDimensions.x,threadDimensions.y); 
		return dim3(0,0);
	}
	
	return dim3(numBlocksPerRow,numRows);
}

/**
* Calculates the djb2 hash.
* @param str The string being hashed.
* @param start The starting point of the word in the array.
* @return Returns the djb2 hash in long format.
*/
__device__ unsigned long djb2Hash(unsigned char* str,int start){
	unsigned long hash = 5381;
	int c;
	while(str[start]!=','){
		c = (int)str[start];
		hash = ((hash<<5)+hash)+c;
		start++;
	}	
	return hash;
}

/**
* Calculates the sdbm hash.
* @param str The string being hashed.
* @param start The starting point of the word in the array.
* @return Returns the sdbm hash in long format.
*/
__device__ unsigned long sdbmHash(unsigned char* str,int start){
	unsigned long hash = 0;
	int c = 0;
	while(str[start]!=','){
		c = (int)str[start];
		hash = c+(hash<<6)+(hash<<16)-hash;
		start++;
	}
	return hash;
}

__device__ int calculateCurrentWord(){
	int numThreadsPrevRows = (blockDim.x*gridDim.x)*blockIdx.y+
														blockDim.x*blockIdx.x;
	return  threadIdx.x+numThreadsPrevRows;
}

__device__ int calculateIndex(char* dev_bloom,int* dev_size,char* dev_words,
	int wordStartingPosition){	

	unsigned long firstValue = djb2Hash((unsigned char*)dev_words,wordStartingPosition)%dev_size[0];	
	unsigned long secondValue = sdbmHash((unsigned char*)dev_words,wordStartingPosition)%dev_size[0];
	secondValue = (secondValue*threadIdx.y*threadIdx.y)%dev_size[0];
	return (firstValue+secondValue)%dev_size[0];		

}


/**
* Responsible for inserting words using the gpu.
* @param dev_bloom The bloom filter being used.
* @param dev_size The size of the bloom filter being used.
* @param dev_words The words being inserted.
* @param dev_positions The starting positions of the words.
* @param dev_numWords The number of words being inserted.
*/
__global__ void insertWordsGpu(char* dev_bloom,int* dev_size,char* dev_words,
	int* dev_positions,int* dev_numWords){
	int currentWord = calculateCurrentWord();
	if(currentWord>=dev_numWords[0])
		return;
	int wordStartingPosition = dev_positions[currentWord]; 	
	int setIdx = calculateIndex(dev_bloom,dev_size,dev_words,
		wordStartingPosition);
	dev_bloom[setIdx]=1;
}

/**
* Responsible for querying words using the gpu.
*/
__global__ void queryWordsGpu(char* dev_bloom,int* dev_size,char* dev_words,
	int* dev_positions,char* dev_results,int* dev_numWords){

	int currentWord = calculateCurrentWord();
	if(currentWord>=dev_numWords[0])
		return;
	int wordStartingPosition = dev_positions[currentWord]; 
	int getIdx = calculateIndex(dev_bloom,dev_size,dev_words,
		wordStartingPosition);
	__syncthreads();
	
	if(dev_bloom[getIdx]==0){
		dev_results[currentWord]=0;
	}
}

/**
* Responsible for inserting words into the bloom filter.
*/
cudaError_t insertWords(char* dev_bloom,int* dev_size,char* dev_words,
	int* dev_offsets,int numWords,int numHashes,int device){

	dim3 threadDimensions = calculateThreadDimensions(numWords,numHashes,device);
	dim3 blockDimensions = calculateBlockDimensions(threadDimensions,numWords,
		device);

	int* dev_numWords = allocateAndCopyIntegers(&numWords,1);
	//Actually insert the words.
	insertWordsGpu<<<blockDimensions,threadDimensions>>>(dev_bloom,dev_size
		,dev_words,dev_offsets,dev_numWords);
	cudaThreadSynchronize();
	freeIntegers(dev_numWords);
	//Check for errorrs...
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		printf("%s \n",cudaGetErrorString(error));
		printf("Dimensions calculated: \n");
		printf("threadDim: %i,%i \n",threadDimensions.x,threadDimensions.y); 
		printf("BlockDim: %i,%i \n",blockDimensions.x,blockDimensions.y);
		return error;
	}
	return cudaSuccess;			 				
}

/**
* Responsible for uerying words inserted into the bloom filter
*/
cudaError_t queryWords(char* dev_bloom,int* dev_size,char* dev_words,
	int* dev_offsets,int numWords,int numHashes,int device,char* dev_result){

	dim3 threadDimensions = calculateThreadDimensions(numWords,numHashes,device);
	dim3 blockDimensions = calculateBlockDimensions(threadDimensions,numWords,
		device);
	
	int* dev_numWords = allocateAndCopyIntegers(&numWords,1);	

	//Actually query the words.
	queryWordsGpu<<<blockDimensions,threadDimensions>>>(dev_bloom,dev_size
		,dev_words,dev_offsets,dev_result,dev_numWords);
	cudaThreadSynchronize();
	freeIntegers(dev_numWords);
	
	//Check for errorrs...
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		printf("%s \n",cudaGetErrorString(error));
		printf("Dimensions calculated: \n");
		printf("threadDim: %i,%i \n",threadDimensions.x,threadDimensions.y); 
		printf("BlockDim: %i,%i \n",blockDimensions.x,blockDimensions.y);
		return error;
	}
	
	return cudaSuccess;			 				
}
