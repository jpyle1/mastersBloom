#include "Hash.h"
#include <cstdio>

/**
* Calculates the djb2 hash.
* @param str The string being hashed.
* @return Returns the djb2 hash in long format.
*/
unsigned long djb2Hash(unsigned char* str){
	unsigned long hash = 5381;
	int c;
	while(c=(int)*str++){
		hash = ((hash<<5)+hash)+c;
	}
	return hash;
}


/**
* Calculates the sdbm hash.
* @param str The string being hashed.
* @return Returns the sdbm hash in long format.
*/
unsigned long sdbmHash(unsigned char* str){
	unsigned long hash = 0;
	int c = 0;
	while(c=(int)*str++)
		hash = c+(hash<<6)+(hash<<16)-hash;
	return hash;
}

/**
* Calculates the djb2 hash.
* @param str The string being hashed.
* @param start The offset of the first letter in the string.
* @return Returns the djb2 hash in long format.
*/
unsigned long djb2HashOffset(char* str,int start){
	unsigned long hash =5381;
	int c;
	for(;str[start]!=',';start++){
		c = (int)str[start];
		hash = ((hash<<5)+hash)+c;
	}
	return hash;
}

/**
* Calculates the sdbm hash
* @param str The string being hashed.
* @param start The offset of the first letter in the string.
* @return Returns the sdbm hash in long format.
*/
unsigned long sdbmHashOffset(char* str,int start){
	unsigned long hash = 0;
	int c = 0;
	for(;str[start]!=',';start++){
		c = (unsigned char)str[start];
		hash = c+(hash<<6)+(hash<<16)-hash; 
	}		
	return hash;
}
