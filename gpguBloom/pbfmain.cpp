#include "../RandomGenerator.h"
#include "../ParseData.h"
#include "Bloom.h"
#include <time.h>
#include <math.h>
/**
*
*/
int main(int argc,char** argv){
	
	//Seed the timer.
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

	int totalNumWords = 0 ;

	//Insert items into the bloom filter.
	int i = 0;
	for(i = 0; i<bloomOptions_t.numBatches;i++){

		//Do we need to insert it multiple times?
		if(i<bloomOptions_t.trueBatches){
			int y = 0;
			for(y = 0;y<bloomOptions_t.numTrueBatchInsertions;y++){
				int randOffset = rand()%2432+10;
				WordAttributes* wordAttributes = loadFile(i);
				if(y == 0)
					totalNumWords+=wordAttributes->numWords;
				insertWordsPBF(dev_bloom,bloomOptions_t.size,
					wordAttributes->currentWords,
					wordAttributes->positions,wordAttributes->numWords,
					wordAttributes->numBytes,bloomOptions_t.numHashes,
					bloomOptions_t.device,bloomOptions_t.prob,randOffset);
				freeWordAttributes(wordAttributes);
			}
		}else{
			//Only insert it once.
			int randOffset = rand()%2432+10;
			WordAttributes* wordAttributes = loadFile(i);
			totalNumWords+=wordAttributes->numWords;
			insertWordsPBF(dev_bloom,bloomOptions_t.size,
				wordAttributes->currentWords,
				wordAttributes->positions,wordAttributes->numWords,
				wordAttributes->numBytes,bloomOptions_t.numHashes,
				bloomOptions_t.device,bloomOptions_t.prob,randOffset);
			freeWordAttributes(wordAttributes);
		}
	}
	
	//Stats.
	int trueNumOnesCounted = 0;
	int falseNumOnesCounted = 0;
	
	FILE* pbfOutput = 0;
	if(bloomOptions_t.pbfOutput){
		pbfOutput = fopen(bloomOptions_t.pbfOutput,"w+");
		if(!pbfOutput){
			printf("Could not open the file for collecting stats \n");
			return 0;
		}
	} 
	
	//Query the words we know to be true...
	i = 0;
	for(;i<bloomOptions_t.trueBatches;i++){
		WordAttributes* wordAttributes = loadFile(i);
		int* results = (int*)calloc(wordAttributes->numWords*sizeof(int),
			sizeof(int));
		queryWordsPBF(dev_bloom,bloomOptions_t.size,wordAttributes->currentWords,
			wordAttributes->positions,wordAttributes->numWords,
			wordAttributes->numBytes,bloomOptions_t.numHashes,bloomOptions_t.device,
			results);

		int x  = 0;
		for(;x<wordAttributes->numWords;x++){
			trueNumOnesCounted+=results[x];
			if(bloomOptions_t.pbfOutput){
				float a = (float)bloomOptions_t.numHashes*totalNumWords*
					bloomOptions_t.prob+bloomOptions_t.size*logf(1-(float)
					results[x]/bloomOptions_t.numHashes); 
				float b = (float)(bloomOptions_t.numHashes-bloomOptions_t.size)*
					bloomOptions_t.prob;
				float f =	a/b;
				a = ((float)bloomOptions_t.numHashes*bloomOptions_t.prob*
					totalNumWords+bloomOptions_t.size*logf(1-(float)results[x]
						/bloomOptions_t.numHashes+(float)1.96*sqrt((1-(float)
						(bloomOptions_t.numHashes-results[x])/bloomOptions_t.numHashes)*
						(bloomOptions_t.numHashes-results[x])/bloomOptions_t.numHashes/
						bloomOptions_t.numHashes)));
				b = (float)(bloomOptions_t.numHashes-bloomOptions_t.size)*
					bloomOptions_t.prob;	
				float f_min = a/b;
				a = ((float)bloomOptions_t.numHashes*bloomOptions_t.prob*
					totalNumWords+bloomOptions_t.size*logf(1-(float)results[x]
						/bloomOptions_t.numHashes-(float)1.96*sqrt((1-(float)
						(bloomOptions_t.numHashes-results[x])/bloomOptions_t.numHashes)*
						(bloomOptions_t.numHashes-results[x])/bloomOptions_t.numHashes/
						bloomOptions_t.numHashes)));
				float f_max =  a/b;
				fprintf(pbfOutput,"%i,%i,%i,%.6f,%.6f,%.6f,\n",i,x,results[x],f,
					f_min,f_max); 
			}
		}	
		free(results);
		freeWordAttributes(wordAttributes);
	}

	//Query the words we know to be false...
	for(i = 0;i<bloomOptions_t.falseBatches;i++){
		
		WordAttributes* wordAttributes = loadFileByPrefix(i,(char*)"q");
		int* results = (int*)calloc(wordAttributes->numWords*sizeof(int),
			sizeof(int));
		
		queryWordsPBF(dev_bloom,bloomOptions_t.size,wordAttributes->currentWords,
			wordAttributes->positions,wordAttributes->numWords,
			wordAttributes->numBytes,bloomOptions_t.numHashes,bloomOptions_t.device,
			results);

		int x = 0;	
		for(;x<wordAttributes->numWords;x++){
			falseNumOnesCounted+=results[x];
			if(bloomOptions_t.pbfOutput){
				float a = (float)bloomOptions_t.numHashes*totalNumWords*
					bloomOptions_t.prob+bloomOptions_t.size*logf(1-(float)
					results[x]/bloomOptions_t.numHashes); 
				float b = (float)(bloomOptions_t.numHashes-bloomOptions_t.size)*
					bloomOptions_t.prob;
				float f =	a/b;
				a = ((float)bloomOptions_t.numHashes*bloomOptions_t.prob*
					totalNumWords+bloomOptions_t.size*logf(1-(float)results[x]
						/bloomOptions_t.numHashes+(float)1.96*sqrt((1-(float)
						(bloomOptions_t.numHashes-results[x])/bloomOptions_t.numHashes)*
						(bloomOptions_t.numHashes-results[x])/bloomOptions_t.numHashes/
						bloomOptions_t.numHashes)));
				b = (float)(bloomOptions_t.numHashes-bloomOptions_t.size)*
					bloomOptions_t.prob;	
				float f_min = a/b;
				a = ((float)bloomOptions_t.numHashes*bloomOptions_t.prob*
					totalNumWords+bloomOptions_t.size*logf(1-(float)results[x]
						/bloomOptions_t.numHashes-(float)1.96*sqrt((1-(float)
						(bloomOptions_t.numHashes-results[x])/bloomOptions_t.numHashes)*
						(bloomOptions_t.numHashes-results[x])/bloomOptions_t.numHashes/
						bloomOptions_t.numHashes)));
				float f_max =  a/b;
				fprintf(pbfOutput,"q%i,%i,%i,%.6f,%.6f,%.6f,\n",i,x,results[x],f,
					f_min,f_max); 


			}
		}	
		
		free(results);
		freeWordAttributes(wordAttributes);
		
	}

	/*	
	printf("TRUE,FALSE,TOTAL %i,%i,%i \n",trueNumOnesCounted,falseNumOnesCounted,
		trueNumOnesCounted+falseNumOnesCounted);
	*/

	if(pbfOutput)
		fclose(pbfOutput);
	
	//Copy the bloom filter to main memory.
	copyCharsToHost(bloom,dev_bloom,bloomOptions_t.size);
	freeChars(dev_bloom);
	//Output the bloom filter
	if(bloomOptions_t.fileName!=0){
		writeBloomFilterToFile(&bloomOptions_t,bloom);
	}	
	free(bloom);
	return 0;
}
