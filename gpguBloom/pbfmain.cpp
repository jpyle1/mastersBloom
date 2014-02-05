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
	if(wasArgSpecified("--help",argv,argc)!=0){
		printHelp();
		return 0;
	}
	
	//Initialize with default configuration.
	BloomOptions_t bloomOptions_t;
	setDefault(&bloomOptions_t);	
	//Parse the user's configuration.
	getConfiguration(&bloomOptions_t,argv,argc);

	if(!wasArgSpecified("--silent",argv,argc)!=0){
		printf("PBF Test\n");
		showDetails(&bloomOptions_t);
	}

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

	//Allocate the GPU bloom filter.
	char* dev_bloom = allocateAndCopyChar(bloom,bloomOptions_t.size); 
	if(dev_bloom==0){
		printf("Could not allocate the bloom filter \n");
		return -1;	
	}

	//Insert items into the bloom filter.
	int i = 0;
	for(i = 0; i<bloomOptions_t.numBatches;i++){
		WordAttributes* wordAttributes = loadFile(i);

		//Insert some batches more frequently than others.
		if(i<bloomOptions_t.trueBatches){
			int x = 0;
			for(;x<bloomOptions_t.numTrueBatchInsertions;x++){
				insertWordsPBF(dev_bloom,bloomOptions_t.size,
					wordAttributes->currentWords,
					wordAttributes->positions,wordAttributes->numWords,
					wordAttributes->numBytes,bloomOptions_t.numHashes,
					bloomOptions_t.device,bloomOptions_t.prob);
			}
		}else{
			insertWordsPBF(dev_bloom,bloomOptions_t.size,
				wordAttributes->currentWords,
				wordAttributes->positions,wordAttributes->numWords,
				wordAttributes->numBytes,bloomOptions_t.numHashes,
				bloomOptions_t.device,bloomOptions_t.prob);
		}
		freeWordAttributes(wordAttributes);
	}

	int trueNumOnesCounted = 0;
	int falseNumOnesCounted = 0;

	//Query the words we know to be true.
	i = 0;
	for(;i<bloomOptions_t.trueBatches;i++){		
		WordAttributes* wordAttributes = loadFile(i);
		int* results = (int*)calloc(wordAttributes->numWords*sizeof(int),
			sizeof(int));
		
		countOnesPBF(dev_bloom,bloomOptions_t.size,wordAttributes->currentWords,
			wordAttributes->positions,wordAttributes->numWords,
			wordAttributes->numBytes,bloomOptions_t.numHashes,bloomOptions_t.device,
			results);
		
		int x = 0;
		for(;x<wordAttributes->numWords;x++){
			trueNumOnesCounted+=results[x];	
		}	

		
		free(results);
		freeWordAttributes(wordAttributes);
	}

	
	//Query the words we know to be false.	
	for(i = 0;i<bloomOptions_t.falseBatches;i++){
		WordAttributes* wordAttributes = loadFileByPrefix(i,(char*)"q");
		int* results = (int*)calloc(wordAttributes->numWords*sizeof(int),
			sizeof(int));

		countOnesPBF(dev_bloom,bloomOptions_t.size,wordAttributes->currentWords,
			wordAttributes->positions,wordAttributes->numWords,
			wordAttributes->numBytes,bloomOptions_t.numHashes,bloomOptions_t.device,
			results);

		int x = 0;
		for(;x<wordAttributes->numWords;x++){
			falseNumOnesCounted+=results[x];	
		}	

		free(results);
		freeWordAttributes(wordAttributes);		
	}
	

	printf("TRUE,FALSE,TOTAL %i %i %i \n",trueNumOnesCounted,falseNumOnesCounted,
		trueNumOnesCounted+falseNumOnesCounted);
	
	//Copy the bloom filter to main memory.	
	copyCharsToHost(bloom,dev_bloom,bloomOptions_t.size);
	//Free the bloom filter.
	freeChars(dev_bloom);	
	//Output the bloom filter.
	if(bloomOptions_t.fileName!=0){
		writeBloomFilterToFile(&bloomOptions_t,bloom);	
	}
	free(bloom);	
	return 0;

}
