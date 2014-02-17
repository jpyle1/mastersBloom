#include "../ParseArgs.h"
#include "../RandomGenerator.h"
#include "../ParseData.h"
#include "Hash.h"
#include <time.h>

/**
* Inserts words into the pbf.
*/
void insertWords(char* bloom,BloomOptions_t* bloomOptions_t,
	WordAttributes* wordAttributes,float prob){
	int numWordsProcessed = 0;
	for(;numWordsProcessed<wordAttributes->numWords;numWordsProcessed++){
		int y = 0;
		for(;y<wordAttributes->numWords;y++){
			unsigned long firstValue = djb2HashOffset(wordAttributes->currentWords,
				wordAttributes->positions[numWordsProcessed])%bloomOptions_t->size;
			unsigned long secondValue = sdbmHashOffset(wordAttributes->currentWords,
				wordAttributes->positions[numWordsProcessed])%bloomOptions_t->size;
			int value = (firstValue+(y*y*secondValue)%bloomOptions_t->size)%
				bloomOptions_t->size;
			float temp = (float)rand()/RAND_MAX;					
			if(temp<prob)	
				bloom[value] = 1;
		}
	}
}

/**
* Responsible for querying the bloom filter.
*/
void queryWords(char* bloom,BloomOptions_t* bloomOptions_t,
	WordAttributes* wordAttributes,int* results){
	int numWordsProcessed = 0;
	for(;numWordsProcessed<wordAttributes->numWords;numWordsProcessed++){
		int y = 0;
		int count = 0;
		for(;y<bloomOptions_t->numHashes;y++){
			unsigned long firstValue = djb2HashOffset(wordAttributes->currentWords,
				wordAttributes->positions[numWordsProcessed])%bloomOptions_t->size;
			unsigned long secondValue = sdbmHashOffset(wordAttributes->currentWords,
				wordAttributes->positions[numWordsProcessed])%bloomOptions_t->size;
			int value = (firstValue+(y*y*secondValue)%bloomOptions_t->size)%
				bloomOptions_t->size;
			count+=bloom[value];
		}				
		results[numWordsProcessed] = count;		
	}
}

/**
* Entry point.
*/
int main(int argc,char** argv){
	//Seed the timer...
	srand(time(0));
		
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
	//Parset he user's configuration
	getConfiguration(&bloomOptions_t,argv,argc);
	//Show the user the configuration.
	if(!wasArgSpecified("--silent",argv,argc)!=0)
		showDetails(&bloomOptions_t);
	//Do we need to generate new test files?
	if(wasArgSpecified("--generate",argv,argc)!=0){
		generateFiles(bloomOptions_t.numBatches,bloomOptions_t.batchSize);
		generateFilesPrefix(bloomOptions_t.falseBatches,bloomOptions_t.batchSize,
			(char*)"q");
		return 0;
	}
	//Create the bloom filter eing used, and initailize with all 0's.
	char* bloom = (char*)calloc(sizeof(char)*bloomOptions_t.size,sizeof(char));
	int i = 0;
	for(i = 0;i<bloomOptions_t.numBatches;i++){
		WordAttributes* wordAttributes = loadFile(i);
		insertWords(bloom,&bloomOptions_t,wordAttributes,bloomOptions_t.prob);
		freeWordAttributes(wordAttributes);
	}

	//Get the stats...
	int numTrueOnesCalculated = 0;
	int numFalseOnesCalculated = 0;
	i = 0;
	for(;i<bloomOptions_t.trueBatches;i++){
		WordAttributes* wordAttributes = loadFile(i);
		int* resultVector = (int*)calloc(sizeof(int)*wordAttributes->numWords,
			sizeof(int));
		queryWords(bloom,&bloomOptions_t,wordAttributes,resultVector);
		int x  = 0;
		for(;x<wordAttributes->numWords;x++){
			numTrueOnesCalculated+=resultVector[x];	
		}
		free(resultVector);
		freeWordAttributes(wordAttributes);
	}

	for(i = 0;i<bloomOptions_t.falseBatches;i++){
		WordAttributes* wordAttributes = loadFileByPrefix(i,(char*)"q");
		int* resultVector = (int*)calloc(sizeof(int)*wordAttributes->numWords,
			sizeof(int));
		queryWords(bloom,&bloomOptions_t,wordAttributes,resultVector);
		int x  = 0;
		for(;x<wordAttributes->numWords;x++){
			numFalseOnesCalculated+=resultVector[x];
		}		
		free(resultVector);
		freeWordAttributes(wordAttributes);
	}	

	printf("TRUE,FALSE,TOTAL %i,%i,%i \n",numTrueOnesCalculated,
		numFalseOnesCalculated,(numTrueOnesCalculated+numFalseOnesCalculated));
		
	if(bloomOptions_t.fileName!=0){
		writeBloomFilterToFile(&bloomOptions_t,bloom);
	}

	free(bloom);

	return 0;
}
