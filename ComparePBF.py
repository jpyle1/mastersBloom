import subprocess,sys
import time

def main(args):	
	#Firstly, vary the number of hash functions.
	for numHash in range(1,3000):
		numHash*=10 
		#Now, vary size (in bytes) of the batches being inserted. 
		for batchSize in range(1,1000):
			batchSize*=10000	
			#Remove any data that already exists in the directory.
			subprocess.call(["rm","-rf",".txt"])
			#Generate the batch data.
			subprocess.call(["bin/pbfmain.exe","--batchSize",str(batchSize),
				"--generate","--silent"])	
			#Setup the basic arguments.
			basicArgs = ["--hashes",str(numHash),"--prob",".002","--batchSize",
				str(batchSize),"--silent","--size","100000000"]
			pbfArgs = ["bin/pbfmain.exe"]
			pbfRun = []
			pbfRun[len(pbfRun):] = pbfArgs
			pbfRun[len(pbfRun):] = basicArgs
			startPbf = time.time()		
			pbfProcess = subprocess.call(pbfRun)
			endPbf = time.time()
			regArgs = ["bin/pbfReg.exe"]
			regRun = []
			regRun[len(regRun):] = regArgs
			regRun[len(regRun):] = basicArgs
			startReg = time.time()
			regProcess = subprocess.call(regRun)
			endReg = time.time()
			line = (str(numHash)+","+str(batchSize)+","+str(endPbf-startPbf)+","+
				str(endReg-startReg))
			print line
if __name__=="__main__":
	main(sys.argv[1:len(sys.argv)])
