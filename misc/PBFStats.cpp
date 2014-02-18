#include "../PBFStats.h"

/**
* Responsible for calculating the stats relating to a pbf.
*/
void writeStats(FILE* pbfOutput,int fileIndex,int* counts,
	int numCounts,int k,float p,int n,int m){

	int i  = 0;
	for(;i<numCounts;i++){
		float a = 0.0f;	
		float b = 0.0f;
		a = (float)k*p*n+m*logf(1-(float)counts[i]/k);
		b = (float)(k-m)*p;
		float f = a/b;
		a = ((float)k*p*n+m*logf(1-(float)counts[i]/k+(float)1.96*sqrt((1-(float)(k-counts[i])/k)*(k-counts[i])/k/k)));
		b = (float)(k-m)*p;
		float fMin = a/b;
		a = ((float)k*p*n+m*logf(1-(float)counts[i]/k-(float)1.96*sqrt((1-(float)(k-counts[i])/k)*(k-counts[i])/k/k)));
		float fMax = a/b;		
		fprintf(pbfOutput,"%i,%i,%i,%.6f,%.6f,%.6f\n",fileIndex,i,counts[i],
			f,fMin,fMax);
	}

}



