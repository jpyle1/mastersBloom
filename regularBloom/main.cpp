#include "../ParseArgs.h"
#include "../RandomGenerator.h"
#include "../ParseData.h"
#include "Hash.h"


/**
* Inserts items into the bloom filter.
* @param bloom The bloom filter in use.
* @param bloomOptions_t The options used in to build the bloom filter. 
* @param wordAttributes Words and stats about the words being inserted.
*/
void insertWords(char* bloom,BloomOptions_t* bloomOptions_t,
	WordAttributes* wordAttributes){
	int numWordsProcessed = 0;
	for(;numWordsProcessed<wordAttributes->numWords;numWordsProcessed++){
		int y = 0;
		for(y = 0; y<bloomOptions_t->numHashes;y++){
			unsigned long firstValue = djb2HashOffset(wordAttributes->currentWords,
				wordAttributes->positions[numWordsProcessed])%bloomOptions_t->size;
			unsigned long secondValue = sdbmHashOffset(wordAttributes->currentWords,
				wordAttributes->positions[numWordsProcessed])%bloomOptions_t->size;
			int value = (firstValue+((y*y)*secondValue)%bloomOptions_t->size)%bloomOptions_t->size;
			bloom[value]=1;	
		}
	}
}

/**
* Responsibe for querying the bloom filter.
*/
void queryWords(char* bloom,BloomOptions_t* bloomOptions_t,
	WordAttributes* wordAttributes,char* resultVector){

	int numWordsProcessed = 0;
	for(;numWordsProcessed<wordAttributes->numWords;numWordsProcessed++){
		int y = 0;
		int isTrue = 1;
		for(y = 0; y<bloomOptions_t->numHashes;y++){
			unsigned long firstValue = djb2HashOffset(wordAttributes->currentWords,
				wordAttributes->positions[numWordsProcessed])%bloomOptions_t->size;
			unsigned long secondValue = sdbmHashOffset(wordAttributes->currentWords,
				wordAttributes->positions[numWordsProcessed])%bloomOptions_t->size;
			int value = (firstValue+((y*y)*secondValue)%bloomOptions_t->size)%bloomOptions_t->size;
			if(bloom[value]!=1){
				isTrue = 0;
				break;
			}
		}
		if(isTrue==1){
			resultVector[numWordsProcessed] = 1;
		}else{
			resultVector[numWordsProcessed] = 0;
		}
	}

}

/**
* Entry point to the program that tests the bloom filter.
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
	//Show the user the configuration.	
	showDetails(&bloomOptions_t);

	//Do we need to generate new test files?
	if(wasArgSpecified("--generate",argv,argc)!=0){		
		generateFiles(bloomOptions_t.numBatches,bloomOptions_t.batchSize);	
	}

	//Create the bloom filter being used, and initialize it with all 0's.
	char* bloom = (char*)malloc(sizeof(char)*bloomOptions_t.size);
	memset(bloom,0,bloomOptions_t.size);	


	//Insert words into the filter.
	int i = 0;
	for(i = 0;i<bloomOptions_t.numBatches;i++){
		WordAttributes* wordAttributes = loadFile(i);
		insertWords(bloom,&bloomOptions_t,wordAttributes);
		freeWordAttributes(wordAttributes);	
	}

	//Query Words
	int numTrue = 0;
	int numCalcTrue = 0;
	int numFalse = 0;
	int numCalcFalse = 0;	
	
	//Query the words that we know should be in the bloom filter.
	i = 0;
	for(i = 0;i<bloomOptions_t.trueBatches;i++){
		WordAttributes* wordAttributes = loadFile(i);
		numTrue+=wordAttributes->numWords;
		char* resultVector = (char*)calloc(sizeof(char)*wordAttributes->numWords,
			sizeof(char));
	
		queryWords(bloom,&bloomOptions_t,wordAttributes,resultVector);		
		int x = 0;
		for(x = 0; x<wordAttributes->numWords;x++){
			if(resultVector[x] == 0){
				numCalcFalse+=1;
			}else{
				numCalcTrue+=1;
			}
		}

		freeWordAttributes(wordAttributes);	
		free(resultVector);	
	}

	//Query words that with a high probability are not in the bloom filter.
	//Firstly, create the false batches.
	//Do we need to generate new test files?
	if(wasArgSpecified("--generate",argv,argc)!=0){		
		generateFilesPrefix(bloomOptions_t.falseBatches,bloomOptions_t.batchSize,
			(char*)"q");
	}

	i = 0;
	for(i = 0;i<bloomOptions_t.falseBatches;i++){
		WordAttributes* wordAttributes = loadFileByPrefix(i,(char*)"q");		
		numFalse += wordAttributes->numWords;
		char* resultVector = (char*)calloc(sizeof(char)*wordAttributes->numWords,
			sizeof(char));
	
		queryWords(bloom,&bloomOptions_t,wordAttributes,resultVector);		
		int x = 0;
		for(x = 0; x<wordAttributes->numWords;x++){
			if(resultVector[x] == 0){
				numCalcFalse+=1;
			}else{
				numCalcTrue+=1;
			}
		}

		freeWordAttributes(wordAttributes);	
		free(resultVector);		
	}

	//Print the query stats.
	printf("calcTrue: %i  KnownTrue: %i \ncalcFalse: %i  KnownFalse: %i\n",
		numCalcTrue,numTrue,numCalcFalse,numFalse);

	if(bloomOptions_t.fileName!=0){
		writeBloomFilterToFile(&bloomOptions_t,bloom);				
	}
	free(bloom);
}
