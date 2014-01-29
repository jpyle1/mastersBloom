
/**
* Contains the interface for using the simple hash functions.
* Algorithms borrowed from cse.yorku.ca.
*/

/**
* Calculates the djb2 hash.
* @param str The string being hashed.
* @return Returns the djb2 hash in long format.
*/
unsigned long djb2Hash(unsigned char* str);

/**
* Calculates the sdbm hash.
* @param str The string being hashed.
* @return Returns the sdbm hash in long format.
*/
unsigned long sdbmHash(unsigned char* str);

/**
* Calculates the djb2 hash.
* @param str The string being hashed.
* @param start The offset of the first letter in the string.
* @return Returns the djb2 hash in long format.
*/
unsigned long djb2HashOffset(char* str,int start);

/**
* Calculates the sdbm hash
* @param str The string being hashed.
* @param start The offset of the first letter in the string.
* @return Returns the sdbm hash in long format.
*/
unsigned long sdbmHashOffset(char* str,int start);
