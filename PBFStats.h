#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/**
* Holds the calculated PBF stats.
*/
typedef struct stats{
	float f;
	float fMin;
	float fMax;	
}PBFStats;


/**
* Responsible for calculating the stats relating to a pbf.
*/
void writeStats(FILE* pbfOutput,int fileIndex,int* counts,int numCounts,
	int k,float p,int n,int m);


