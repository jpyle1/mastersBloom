#include "../ParseData.h"

/**
* Turns a double array of character pointers into a single array
* of characters separated by commas.
* @param words The words being turned into CSV format.
* @param numWords The number of words in total.
* @return Returns the words in a single, csv array.
*/
char* toCSV(char** words,int numWords){
	if(numWords<=0){
		return 0;
	}
	int firstWordLength = strlen(words[numWords-1]);
	char* singleArray = (char*)malloc(sizeof(char)*firstWordLength+sizeof(char));
	memset(singleArray,0,firstWordLength+1);
	memcpy(singleArray,words[numWords-1],firstWordLength);
	singleArray[firstWordLength] = ',';
	numWords-=2;
	while(numWords>=0){
		int wordLength = strlen(words[numWords]);
		singleArray = (char*)realloc(singleArray,sizeof(char)*wordLength+1);
		strcat(singleArray,words[numWords]);
		strcat(singleArray,",");	 
		numWords--;	
	}	
	return singleArray;
}


/**
* Parses the WordAttributes of a file.
* @param fileData The data located inside the file.
* @param fileSize The number of bytes in the file.
* @return The WordAttributes that describes the data.
*/
WordAttributes* parseWordAttributes(char* fileData,int fileSize){
	char* fileDataDeepCopy = (char*)malloc(sizeof(char)*fileSize);
	memset(fileDataDeepCopy,0,fileSize);
	memcpy(fileDataDeepCopy,fileData,sizeof(char)*fileSize);
	
	//Holds the data structure describing the words that will be returned.
	WordAttributes* wordAttributes = (WordAttributes*)malloc(sizeof(WordAttributes));

	//Number of words that have currently been read.
	int words = 0;	
	
	//Create the list of words that have been read.
	char* newWord = strtok(fileData,",");
	int currentStartPosition = 0;
	int currentEndPosition = strlen(newWord);

	//Get the individual starting and stopping position of each word.
	Positions* currentPosition = (Positions*)malloc(sizeof(Positions));	
	Positions* firstPosition = currentPosition;	
	currentPosition->currentStartPosition = currentStartPosition;
	currentPosition->currentEndPosition = currentEndPosition;		
	currentPosition->nextPosition = 0;
	
	while(newWord!=0){
		//Increment the number of words added.
		words++;	

		newWord = strtok(0,",");
		if(newWord == 0){
			break;				
		}
		//Update the current start and end position.
		currentStartPosition = currentEndPosition+1;
		currentEndPosition = currentStartPosition+strlen(newWord);
		//Set the positions.
		Positions* nextPosition = (Positions*)malloc(sizeof(Positions));
		nextPosition->currentStartPosition = currentStartPosition;
		nextPosition->currentEndPosition = currentEndPosition;		
		//Update the current position.
		currentPosition->nextPosition = nextPosition;
		currentPosition = nextPosition;
		currentPosition->nextPosition = 0;
	}
	wordAttributes->currentWords = fileDataDeepCopy;
	wordAttributes->numWords = words;
	wordAttributes->positions = cleanAndGetPositions(firstPosition,words);
	wordAttributes->numBytes = fileSize;
	return wordAttributes;

}



/**
* Loads files specified by name.
* @param fileName The name of the file being opened.
* @return Returns a WordAttributes structure describing the words that were parsed.
*/
WordAttributes* loadFileByName(char* newFile){
	//Load the file.
	FILE* currentFile = fopen(newFile,"r");
	if(!currentFile){
		return 0;
	}
	//Get the file size.
	fseek(currentFile,0L,SEEK_END);	
	int fileSize = (int)ftell(currentFile);
	fseek(currentFile,0L,SEEK_SET);
	//Add one for the null terminating characte for string.s
	fileSize+=1;

	char* fileData = (char*)malloc(sizeof(char)*fileSize);
	memset(fileData,0,fileSize);
	fread(fileData,sizeof(char),fileSize,currentFile);

	//We are done with the file, so free the resources.
	free(newFile);
	fclose(currentFile);

	parseWordAttributes(fileData,fileSize);
}



/**
* Loads a file.
* @param index The index of the file being loaded.
* @return Returns a structure describing the loaded words.
*/
WordAttributes* loadFile(int index){
	//Allocate the name of the file in memory.
	char* newFile = (char*)malloc(sizeof(char)*FILENAME_MAX);	
	//Create the file name based on the index.
	sprintf(newFile,"%i",index);
	strcat(newFile,".txt");
	return loadFileByName(newFile);
}

/**
* Loads files by prefix
* @param index The index of the file being loaded.
* @param prefix The prefix to the filename.
* @return Returns a WordAttributes structure desribing the words that were parsed. 
*/
WordAttributes* loadFileByPrefix(int index,char* prefix){
	//Allocate the name of the file in memory.
	char* newFile = (char*)malloc(sizeof(char)*FILENAME_MAX);	
	//Create the file name based on the index.
	sprintf(newFile,"%s%i",prefix,index);
	strcat(newFile,".txt");
	return loadFileByName(newFile);
}


/**
* Converts the list of Positions to an array of integers and garbage collects the list.
* @firstPosition The first position in the list of positions.
* @numWords The number of words specified.
* @return Returns an array of integers corresponding to the different positions.
*/
int* cleanAndGetPositions(Positions* firstPosition,int numWords){
	//There are two positions for each word.
	int* positions = (int*)malloc(sizeof(int)*numWords);
	Positions* currentPosition = firstPosition;
	int currentIdx = 0;
	for(currentIdx = 0; currentIdx<numWords;currentIdx++){
		positions[currentIdx] = currentPosition->currentStartPosition;
		Positions* oldPosition = currentPosition;
		currentPosition = currentPosition->nextPosition;
		free(oldPosition);
	}
	return positions;
}

/**
* Frees the memory allocated by the WordAttributes.
* @param wordAttributes A pointer to the word attributes being freed.
*/
void freeWordAttributes(WordAttributes* wordAttributes){
	free(wordAttributes->positions);
	free(wordAttributes->currentWords);
	free(wordAttributes);
}

/**
* Responsible for splitting a WordAttributes into two WordAttributes.
*/
WordAttributes* splitWordAttributes(WordAttributes* original){
	return 0;
}

