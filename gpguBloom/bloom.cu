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
		printf("Could not allocate memory for the integers");
		return 0;
	}	 
	result = cudaMemcpy(dev_int,array,sizeof(int)*length,
		cudaMemcpyHostToDevice);
	if(result!=cudaSuccess){
		printf("Could not copy the integers ot the device \n");
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
		printf("Could not allocate the char array \n");
		return 0;
	}	 
	result = cudaMemcpy(dev_array,array,sizeof(char)*length,
		cudaMemcpyHostToDevice);
	if(result!=cudaSuccess){	
		printf("Could copy the char array to the device \n");
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
* Copies an integer array to the host.
*/
cudaError_t copyIntegersToHost(int* array,int* dev_array,int length){
	return cudaMemcpy(array,dev_array,sizeof(int)*length,cudaMemcpyDeviceToHost);
}


/**
* Responsible for calculating the dimenions of the gpu layout being used.
* @param numWords
* @param numHash
* @param device
*/
dim3 calculateThreadDimensions(int numWords,int numHash,
	cudaDeviceProp* deviceProps){
	if(numWords == 0 || numHash == 0){
		printf("Nothing to do \n");
		return dim3(0,0,0);
	}
		
	//Firstly, solve for the max number of words that 
	//Can be processed in one thread block.
	int maxWordPerBlock = deviceProps->maxThreadsPerBlock/numHash;

	//Check to see if the user demanded more hash functions than
	//A single block can support. If so, only one word per block
	//Will be processed.	
	if(maxWordPerBlock ==0){
		maxWordPerBlock = 1;
		numHash = deviceProps->maxThreadsPerBlock;
	}
	//Try to group the words into sets of 32.
	int wordsPerBlock = 32*(maxWordPerBlock/32);
	if(wordsPerBlock ==0)
		wordsPerBlock = maxWordPerBlock;
	//If all the words can fit in one block.
	if(numWords<=maxWordPerBlock)
		wordsPerBlock = numWords;	
	dim3 threadDimensions(wordsPerBlock,numHash);
	return threadDimensions;
}

/**
* Responsible for calculating the thread dimensions of the gpu layout.
* @param threadDimensions the dimensions of the thread block.
* @param device The id of the device being used.
*/
dim3 calculateBlockDimensions(dim3 threadDimensions,int numWords,int numHash,
	cudaDeviceProp* deviceProps){
	if(numWords == 0){
		printf("Nothing to do \n");
		return dim3(0,0,0);
	}

	//Calculate the number of blocks needed to process all of the words.
	int numBlocksNeeded = numWords/threadDimensions.x;
	if(numWords%threadDimensions.x!=0)
		numBlocksNeeded++;

	//Hard coded due to hydra glitch.
	int maxGridSizeX = 65535;
	int numBlocksPerRow;	

	//If we only need part of the first row...	
	if(numBlocksNeeded<=maxGridSizeX)
		numBlocksPerRow = numBlocksNeeded;
	//If we need one or more rows...
	else{
		numBlocksPerRow = maxGridSizeX;
	}
	//Calculate the number of rows needed.		
	int numRows = numBlocksNeeded/numBlocksPerRow;
	if(numBlocksNeeded%numBlocksPerRow!=0){
		numRows++;
	}

	//Add rows for extra hash functions > 1024.	
	numRows = numRows*(numHash/deviceProps->maxThreadsPerBlock)+ 
		numRows*(numHash%deviceProps->maxThreadsPerBlock>0 ? 1 : 0);	
	
	if(numRows>deviceProps->maxGridSize[1]){
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

__device__ int calculateCurrentWord(int numRowsPerHash){
	int numThreadsPrevRows = (blockDim.x*gridDim.x)*(blockIdx.y/numRowsPerHash)+
		blockDim.x*blockIdx.x;
	return  threadIdx.x+numThreadsPrevRows;
}

__device__ int calculateIndex(char* dev_bloom,int size,char* dev_words,
	int wordStartingPosition,int numHash,int numRowsPerHash){	

	unsigned long firstValue = djb2Hash((unsigned char*)dev_words,wordStartingPosition)%size;	
	unsigned long secondValue = sdbmHash((unsigned char*)dev_words,wordStartingPosition)%size;
	int fy = ((blockIdx.y%numRowsPerHash)*(blockDim.y-1)+threadIdx.y);
	if(fy>=numHash)
		return -1;
	secondValue = (secondValue*fy*fy)%size;
	return (firstValue+secondValue)%size;
}

/**
* Multiply with carry. Psuedo random number generation.
* @param m_w The first seed.
* @param m_z The second seed.
*/
__device__ unsigned int get_random(unsigned long m_w,unsigned long m_z){
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	return (unsigned int)((m_z << 16) + m_w);
} 

/**
* Responsible for inserting words using the gpu.
* @param dev_bloom The bloom filter being used.
* @param dev_size The size of the bloom filter being used.
* @param dev_words The words being inserted.
* @param dev_positions The starting positions of the words.
* @param dev_numWords The number of words being inserted.
* @param numHashes The number of hash functions used.
*/
__global__ void insertWordsGpu(char* dev_bloom,int size,char* dev_words,
	int* dev_positions,int numWords,int numHashes,int numRowsPerHash){
	int currentWord = calculateCurrentWord(numRowsPerHash);
	if(currentWord>=numWords)
		return;
	int wordStartingPosition = dev_positions[currentWord]; 	
	int setIdx = calculateIndex(dev_bloom,size,dev_words,
		wordStartingPosition,numHashes,numRowsPerHash);
	if(setIdx<0)
		return;
	dev_bloom[setIdx]=1;
}

/**
* Responsible for inserting words using the gpu.
* @param dev_bloom The bloom filter being used.
* @param dev_size The size of the bloom filter being used.
* @param dev_words The words being inserted.
* @param dev_positions The starting positions of the words.
* @param dev_numWords The number of words being inserted.
* @param numHashes The number of hash functions used.
*/
__global__ void insertWordsGpuPBF(char* dev_bloom,int size,char* dev_words,
	int* dev_positions,int numWords,int numHashes,int numRowsPerHash,float prob){
	int currentWord = calculateCurrentWord(numRowsPerHash);
	if(currentWord>=numWords)
		return;
	int wordStartingPosition = dev_positions[currentWord]; 	
	int setIdx = calculateIndex(dev_bloom,size,dev_words,
		wordStartingPosition,numHashes,numRowsPerHash);
	int fy = ((blockIdx.y%numRowsPerHash)*(blockDim.y-1)+threadIdx.y);
	clock_t currentTime = clock();
	unsigned int randVal = 
		get_random(currentTime,(unsigned long)(currentTime*(fy+1)*(threadIdx.x+1)));
	float calcProb = (float)randVal*2.5f/(UINT_MAX);
	//If the number of hash functions was exceeded.
	if(setIdx<0)
		return;
	if(dev_bloom[setIdx]==0 && calcProb<prob){
		dev_bloom[setIdx]=1;
	}
}


/**
* Responsible for querying words using the gpu.
*/
__global__ void queryWordsGpu(char* dev_bloom,int size,char* dev_words,
	int* dev_positions,char* dev_results,int numWords,int numHashes,
	int numRowsPerHash){

	int currentWord = calculateCurrentWord(numRowsPerHash);
	if(currentWord>=numWords)
		return;

	int wordStartingPosition = dev_positions[currentWord]; 
	int getIdx = calculateIndex(dev_bloom,size,dev_words,
		wordStartingPosition,numHashes,numRowsPerHash);
	if(getIdx<0)
		return;
	__syncthreads();	
	if(dev_bloom[getIdx]==0){
		dev_results[currentWord]=0;
	}
}

/**
* Responsible for querying words using the gpu.
*/
__global__ void queryWordsGpuPBF(char* dev_bloom,int size,char* dev_words,
	int* dev_positions,int* dev_results,int numWords,int numHashes,
	int numRowsPerHash){

	int currentWord = calculateCurrentWord(numRowsPerHash);
	if(currentWord>=numWords)
		return;

	int wordStartingPosition = dev_positions[currentWord]; 
	int getIdx = calculateIndex(dev_bloom,size,dev_words,
		wordStartingPosition,numHashes,numRowsPerHash);
	if(getIdx<0)
		return;
	atomicAdd(&dev_results[currentWord],dev_bloom[getIdx]);
}


/**
* Responsible for inserting words into the bloom filter.
*/
cudaError_t insertWords(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device){

	//Get the device information being used.
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps,device);

	//Calculate the dimensions needed.
	dim3 threadDimensions = calculateThreadDimensions(numWords,numHashes,
		&deviceProps);
	dim3 blockDimensions = calculateBlockDimensions(threadDimensions,numWords,
		numHashes,&deviceProps);
	//Calculate the number of extra rows to calculate hashes >1024.
	int numRowPerHash = numHashes/deviceProps.maxThreadsPerBlock + 
		(numHashes%deviceProps.maxThreadsPerBlock>0 ? 1 : 0);

	//Allocate the information.
	int* dev_offsets = allocateAndCopyIntegers(offsets,numWords);
	if(!dev_offsets){
		return cudaGetLastError();
	}
		
	char* dev_words = allocateAndCopyChar(words,numBytes);
	if(!dev_words){
		return cudaGetLastError();
	}

	//Actually insert the words.
	insertWordsGpu<<<blockDimensions,threadDimensions>>>(dev_bloom,size
		,dev_words,dev_offsets,numWords,numHashes,numRowPerHash);
	cudaThreadSynchronize();

	//Check for errorrs...
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		printf("%s \n",cudaGetErrorString(error));
		printf("Dimensions calculated: \n");
		printf("threadDim: %i,%i \n",threadDimensions.x,threadDimensions.y); 
		printf("BlockDim: %i,%i \n",blockDimensions.x,blockDimensions.y);
		return error;
	}

	if(!freeChars(dev_words) || !freeIntegers(dev_offsets))
		return cudaGetLastError();

	return cudaSuccess;			 				
}

/**
* Responsible for inserting words into the PBF bloom filter.
*/
cudaError_t insertWordsPBF(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device,float prob){

	//Get the device information being used.
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps,device);

	//Calculate the dimensions needed.
	dim3 threadDimensions = calculateThreadDimensions(numWords,numHashes,
		&deviceProps);
	dim3 blockDimensions = calculateBlockDimensions(threadDimensions,numWords,
		numHashes,&deviceProps);
	//Calculate the number of extra rows to calculate hashes >1024.
	int numRowPerHash = numHashes/deviceProps.maxThreadsPerBlock + 
		(numHashes%deviceProps.maxThreadsPerBlock>0 ? 1 : 0);

	//Allocate the information.
	int* dev_offsets = allocateAndCopyIntegers(offsets,numWords);
	if(!dev_offsets){
		return cudaGetLastError();
	}
		
	char* dev_words = allocateAndCopyChar(words,numBytes);
	if(!dev_words){
		return cudaGetLastError();
	}

	//Actually insert the words.
	insertWordsGpuPBF<<<blockDimensions,threadDimensions>>>(dev_bloom,size
		,dev_words,dev_offsets,numWords,numHashes,numRowPerHash,prob);
	cudaThreadSynchronize();

	//Check for errorrs...
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		printf("%s \n",cudaGetErrorString(error));
		printf("Dimensions calculated: \n");
		printf("threadDim: %i,%i \n",threadDimensions.x,threadDimensions.y); 
		printf("BlockDim: %i,%i \n",blockDimensions.x,blockDimensions.y);
		return error;
	}

	if(!freeChars(dev_words) || !freeIntegers(dev_offsets))
		return cudaGetLastError();

	return cudaSuccess;			 				

}

/**
* Responsible for uerying words inserted into the bloom filter
*/
cudaError_t queryWords(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device,
	char* results){

	//Get the device information being used.
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps,device);

	dim3 threadDimensions = calculateThreadDimensions(numWords,numHashes,
		&deviceProps);
	dim3 blockDimensions = calculateBlockDimensions(threadDimensions,numWords,
		numHashes,&deviceProps);

	int numRowPerHash = numHashes/deviceProps.maxThreadsPerBlock + 
		(numHashes%deviceProps.maxThreadsPerBlock>0 ? 1 : 0);

	int* dev_offsets = allocateAndCopyIntegers(offsets,numWords);
	if(!dev_offsets){
		return cudaGetLastError();
	}
		
	char* dev_words = allocateAndCopyChar(words,numBytes);
	if(!dev_words){
		return cudaGetLastError();
	}

	char* dev_results = allocateAndCopyChar(results,numWords);
	if(!dev_results){
		return cudaGetLastError();
	}

	//Actually query the words.
	queryWordsGpu<<<blockDimensions,threadDimensions>>>(dev_bloom,size
		,dev_words,dev_offsets,dev_results,numWords,numHashes,
		numRowPerHash);
	cudaThreadSynchronize();
		
	//Check for errorrs...
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		printf("%s \n",cudaGetErrorString(error));
		printf("Dimensions calculated: \n");
		printf("threadDim: %i,%i \n",threadDimensions.x,threadDimensions.y); 
		printf("BlockDim: %i,%i \n",blockDimensions.x,blockDimensions.y);
		return error;
	}

	if(!copyCharsToHost(results,dev_results,numWords) ||
		!freeChars(dev_words) ||
		!freeChars(dev_results) ||
		!freeIntegers(dev_offsets))
			return cudaGetLastError();

	return cudaSuccess;			 				
}

cudaError_t queryWordsPBF(char* dev_bloom,int size,char* words,
	int* offsets,int numWords,int numBytes,int numHashes,int device,
		int* results){

	//Get the device information being used.
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps,device);

	dim3 threadDimensions = calculateThreadDimensions(numWords,numHashes,
		&deviceProps);
	dim3 blockDimensions = calculateBlockDimensions(threadDimensions,numWords,
		numHashes,&deviceProps);

	int numRowPerHash = numHashes/deviceProps.maxThreadsPerBlock + 
		(numHashes%deviceProps.maxThreadsPerBlock>0 ? 1 : 0);

	int* dev_offsets = allocateAndCopyIntegers(offsets,numWords);
	if(!dev_offsets){
		return cudaGetLastError();
	}
		
	char* dev_words = allocateAndCopyChar(words,numBytes);
	if(!dev_words){
		return cudaGetLastError();
	}

	int* dev_results = allocateAndCopyIntegers(results,numWords);
	if(!dev_results){
		return cudaGetLastError();
	}

	//Actually query the words.
	queryWordsGpuPBF<<<blockDimensions,threadDimensions>>>(dev_bloom,size
		,dev_words,dev_offsets,dev_results,numWords,numHashes,
		numRowPerHash);
	cudaThreadSynchronize();
		
	//Check for errorrs...
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess){
		printf("%s \n",cudaGetErrorString(error));
		printf("Dimensions calculated: \n");
		printf("threadDim: %i,%i \n",threadDimensions.x,threadDimensions.y); 
		printf("BlockDim: %i,%i \n",blockDimensions.x,blockDimensions.y);
		return error;
	}

	if(!copyIntegersToHost(results,dev_results,numWords) ||
		!freeChars(dev_words) ||
		!freeIntegers(dev_results) ||
		!freeIntegers(dev_offsets))
			return cudaGetLastError();

	return cudaSuccess;			 				
}



