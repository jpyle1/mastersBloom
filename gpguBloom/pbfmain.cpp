#include "../RandomGenerator.h"
#include "../ParseData.h"
#include "Bloom.h"
#include <time.h>
/**
*
*/
int main(int argc,char** argv){
	
	//Seed the timer.
	srand(time(0));
	int randOffset = rand()%2432+10;
	for(int i = 0; i<randOffset;i++){
	}		


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

	if(!wasArgSpecified("--silent",argv,argc))
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
		printf("%i \n",wordAttributes->numWords);
		insertWordsPBF(dev_bloom,bloomOptions_t.size,wordAttributes->currentWords,
			wordAttributes->positions,wordAttributes->numWords,
			wordAttributes->numBytes,bloomOptions_t.numHashes,
			bloomOptions_t.device,bloomOptions_t.prob);
		freeWordAttributes(wordAttributes);
	}


}
