#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/**
* Prints the help about the program.
* 
*/
void printHelp();

/**
* Determines if a particular argument was specified.
* @param argument The argument being looked for.
* @param args The arguments the user specified.
* @param argc The number of arguments the user specified.
*/
int wasArgSpecified(const char* argument,char** args,int argc);

/**
* Returns the value of the argument..
* @param argument The argument being searched for.
* @param args The argument list specified by the user.
* @param argc The number of arguments specifeid.
* @return Returns the specified value of the argument if it exists.
*/
char* getArgValue(const char* argument,char** args,int argc);

/**
* Responsible for holding information about the bloom filter.
* 
*/
typedef struct BloomOptions{
	/**
	* Holds the number of bits being used in the bloom filter.
	*/
	int size;

	/**
	* Holds the number of hash functions that should be specified.
	*/
	int numHashes;

	/**
	* Holds the size of the batch to be inserted into the filter.
	*/
	int batchSize;

	/**
	* Holds the number of batches to be inserted into the filter.
	*/
	int numBatches;

	/**
	* Determines the number of times to insert the number of true batches.
	*/
	int numTrueBatchInsertions;

	/**
	* Holds the number of true batches that should be used when querying.
	*/
	int trueBatches;

	/**
	* Holds the number of false batches that should be used when querying.
	*/
	int falseBatches;

	/**
	* Determines the device being used. (Only for cuda)
	*/
	int device;

	/**
	* Determines the filename where the bloom filter should be outputted to.
	*/
	char* fileName;

	/**
	* Determines the float of the probability being used.
	*/
	float prob;

} BloomOptions_t;

/**
* Sets the default parameters of the bloom filter (if a parameter is not specified this value will be used
* @param bloomOptions A pointer to the bloom filter being used.
*/
void setDefault(BloomOptions_t* bloomOptions);

/**
* Parse the bloom filter options specified by the user.
* @param bloomOptions The options to be created.
* @param args The arguments specified by the user.
* @param argc The number of arguments specified by the user. 
*/
void getConfiguration(BloomOptions_t* bloomOptions,char** args,int argc);

/**
* Show the detail of the bloom filter.
*/ 
void showDetails(BloomOptions_t* bloomOptions);

/**
* Responsible for writing the bloom filter to a file.
* @param *bloomOptions A pointer to the options used to describe the bloom filter.
* @param *bloom A pointer to the bloom filter. 
*/
void writeBloomFilterToFile(BloomOptions_t* bloomOptions,char* bloom);
