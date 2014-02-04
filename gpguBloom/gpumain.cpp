#include "../RandomGenerator.h"
#include "../ParseData.h"
#include "Bloom.h"

/**
*
*/
int main(int argc,char** argv){
	/**
	* Does the user need help?
	*/ 
	if(wasArgSpecified("--help",argv,argc)!=0)
		printHelp();
	
	//Initialize with default configuration.
	BloomOptions_t bloomOptions_t;
	setDefault(&bloomOptions_t);	
	//Parse the user's configuration.
	getConfiguration(&bloomOptions_t,argv,argc);

	if(!wasArgSpecified("--silent",argv,argc)!=0)
		showDetails(&bloomOptions_t);

	//Do we need to generate new test files?
	if(wasArgSpecified("--generate",argv,argc)!=0){	
		//Generate the file to be inserted.	
		generateFiles(bloomOptions_t.numBatches,bloomOptions_t.batchSize);
		//Generate fale data that is queried.
		generateFilesPrefix(bloomOptions_t.falseBatches,bloomOptions_t.batchSize,
			(char*)"q");
		return 0;
	}

	//Create the bloom filter being used, and initialize it with all 0's.
	char* bloom = (char*)malloc(sizeof(char)*bloomOptions_t.size);
	memset(bloom,0,bloomOptions_t.size);
	//Create an array holding the host size;
	int* bloomSize = &bloomOptions_t.size; 

	//Allocate the GPU bloom filter.
	char* dev_bloom = allocateAndCopyChar(bloom,bloomOptions_t.size); 
	if(dev_bloom==0){
		printf("Could not allocate the bloom filter \n");
		return -1;	
	}
	//Allocate and Copy the size of the bloom filter
	int* dev_size = allocateAndCopyIntegers(bloomSize,1);
	if(dev_size==0){
		printf("Could not allocate the bloom filter size \n");
		return -1;
	}

	//Insert items into the bloom filter.
	int i = 0;
	for(i = 0; i<bloomOptions_t.numBatches;i++){
		WordAttributes* wordAttributes = loadFile(i);
		int* dev_offsets = allocateAndCopyIntegers(wordAttributes->positions,
			wordAttributes->numWords);
		char* dev_words = allocateAndCopyChar(wordAttributes->currentWords,
			wordAttributes->numBytes);
		insertWords(dev_bloom,dev_size,dev_words,dev_offsets,
			wordAttributes->numWords,bloomOptions_t.numHashes,bloomOptions_t.device);
		freeChars(dev_words);
		freeIntegers(dev_offsets);
		freeWordAttributes(wordAttributes);
	}

	//Query Words
	int numTrue = 0;
	int numCalcTrue = 0;
	int numFalse = 0;
	int numCalcFalse = 0;	

	//Query the words we know to be true.
	i = 0;
	for(;i<bloomOptions_t.trueBatches;i++){
		
		WordAttributes* wordAttributes = loadFile(i);
		numTrue += wordAttributes->numWords;
		int* dev_offsets = allocateAndCopyIntegers(wordAttributes->positions,
			wordAttributes->numWords);
		char* dev_words = allocateAndCopyChar(wordAttributes->currentWords,
			wordAttributes->numBytes);
		
		char* resultVector = (char*)malloc(sizeof(char)*wordAttributes->numWords);
		memset(resultVector,1,wordAttributes->numWords);

		char* dev_results = allocateAndCopyChar(resultVector,
			wordAttributes->numWords);
		
		queryWords(dev_bloom,dev_size,dev_words,dev_offsets,
			wordAttributes->numWords,bloomOptions_t.numHashes,bloomOptions_t.device,
			dev_results);
		copyCharsToHost(resultVector,dev_results,wordAttributes->numWords);
		
		int x = 0;
		for(x = 0; x<wordAttributes->numWords;x++){
			if(resultVector[x]==1)
				numCalcTrue+=1;
			else
				numCalcFalse+=1;
		}
		
		free(resultVector);
		freeChars(dev_words);
		freeChars(dev_results);
		freeIntegers(dev_offsets);
	}
	
	for(i = 0;i<bloomOptions_t.falseBatches;i++){
		WordAttributes* wordAttributes = loadFileByPrefix(i,(char*)"q");		
		numFalse += wordAttributes->numWords;
		int* dev_offsets = allocateAndCopyIntegers(wordAttributes->positions,
			wordAttributes->numWords);
		char* dev_words = allocateAndCopyChar(wordAttributes->currentWords,
			wordAttributes->numBytes);
		char* resultVector = (char*)malloc(sizeof(char)*wordAttributes->numWords);
		memset(resultVector,1,wordAttributes->numWords);
		
		char* dev_results = allocateAndCopyChar(resultVector,
			wordAttributes->numWords);
		
		queryWords(dev_bloom,dev_size,dev_words,dev_offsets,
			wordAttributes->numWords,bloomOptions_t.numHashes,bloomOptions_t.device,
			dev_results);
		
		copyCharsToHost(resultVector,dev_results,wordAttributes->numWords);

		int x = 0;
		for(x = 0; x<wordAttributes->numWords;x++){
			if(resultVector[x] == 0){
				numCalcFalse+=1;
			}else{
				numCalcTrue+=1;
			}
		}
		
		free(resultVector);
		freeChars(dev_words);
		freeChars(dev_results);
		freeIntegers(dev_offsets);
		
	}
	

	//Print the query stats.
	printf("calcTrue: %i  KnownTrue: %i \ncalcFalse: %i  KnownFalse: %i\n",
		numCalcTrue,numTrue,numCalcFalse,numFalse);



	//Copy the bloom filter to main memory.	
	copyCharsToHost(bloom,dev_bloom,bloomOptions_t.size);
	//Free the bloom filter.
	freeChars(dev_bloom);	
	freeIntegers(dev_size);
	//Output the bloom filter.
	if(bloomOptions_t.fileName!=0){
		writeBloomFilterToFile(&bloomOptions_t,bloom);	
	}
	free(bloom);	
	return 0;

}
