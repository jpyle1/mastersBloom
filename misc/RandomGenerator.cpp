#include "../RandomGenerator.h"

/**
* Responsible for creating random data of specified sizes.
* @param numBytes The length of the String in bytes.
*/
char* generateRandomString(int numBytes){
	char* randomString = (char*)malloc(numBytes);
	int i =0;
	for(i =0; i<numBytes-1;i++){
		randomString[i] = random()%57+65;
	}
	randomString[numBytes-1]=',';
	return randomString;
}

/**
* Generate file with random strings.
* @param fileName the name of the file being generated
* @param numBytes the number of bytes in the file.
* @return Returns the number of words written.
*/
int generateFile(char* fileName,int numBytes){
	int totalBytesAdded = 0;
	int numWords = 0;
	char* fileBuffer = (char*)malloc(sizeof(char)*numBytes); 
	while(1){
		int numBytesString = random()%50+2;
		char* randomString = generateRandomString(numBytesString);
		if(totalBytesAdded+numBytesString>=numBytes){
			free(randomString);
			break;
		}
		memcpy(fileBuffer+totalBytesAdded,randomString,numBytesString);
		free(randomString);
		totalBytesAdded+=numBytesString;
		numWords++;
	}
	FILE* generatedFile = fopen(fileName,"w");
	fwrite(fileBuffer,sizeof(char),totalBytesAdded,generatedFile);
	fclose(generatedFile);	
	free(fileBuffer);
	return numWords;
}

/**
* Generate files with random Strings. 
* @param bloomOptions_t The bloom options used to create the filter.
*/
void generateFiles(int numFiles, int batchSize){
	//Seed the timer
	srand(time(0));
	int i = 0;
	for(i = 0; i<numFiles; i++){
		char* newFile = (char*)malloc(sizeof(char)*FILENAME_MAX);
		sprintf(newFile,"%i",i);
		strcat(newFile,".txt");
		generateFile(newFile,batchSize);
		free(newFile);	
	}
}

/**
* Generates files with a prefix filename.
*/
void generateFilesPrefix(int numFiles,int batchSize,char* prefix){
	//Seed the timer
	int i = 0;
	for(i = 0; i<numFiles; i++){
		char* newFile = (char*)malloc(sizeof(char)*FILENAME_MAX);
		sprintf(newFile,"%s%i",prefix,i);
		strcat(newFile,".txt");
		generateFile(newFile,batchSize);
		free(newFile);	
	}
}
