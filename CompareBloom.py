import subprocess,sys,optparse,time

def main(hashStart,hashEnd,hashInc,
	batchStart,batchEnd,batchInc):
	#Vary the number of hash functions...
	for numHash in range(hashStart,hashEnd+hashInc,hashInc):
		f=open("data/reg/time/numHash_"+str(numHash),"w+")
		f.write("#Hash,BatchSize(Bytes),GPUTime,RegTime\n")
		#Now, vary size (in bytes) of the batches being inserted.
		for batchSize in range(batchStart,batchEnd+batchInc,batchInc):
			#Remove any data that already exists in the directory.
			subprocess.call(["rm","-rf",".txt"])
			#Generate the batch data.	
			subprocess.call(["bin/main.exe","--batchSize",str(batchSize),
				"--generate","--silent","--numBatches","1","--trueBatches","1",
				"--falseBatches","1"])
			#Setup the basic arguments.
			basicArgs = ["--hashes",str(numHash),"--batchSize",str(batchSize),
				"--silent","--size","100000000","--numBatches","1","--trueBatches","1"
				,"--falseBatches","1"]
			gpuArgs = ["bin/gpuMain.exe"]
			gpuRun = []
			gpuRun[len(gpuRun):] = gpuArgs
			gpuRun[len(gpuRun):] = basicArgs
			#Add one more argument for the output of the bloom filter.
			gpuRun[len(gpuRun):] = ["--file",
				"data/reg/result/gpu.txt"]
			startGpu = time.time()
			gpuProcess = subprocess.call(gpuRun)
			endGpu = time.time()
			regArgs = ["bin/main.exe"]
			regRun = []
			regRun[len(regRun):] = regArgs
			regRun[len(regRun):] = basicArgs
			regRun[len(regRun):] = ["--file",
				"data/reg/result/reg.txt"]
			startReg = time.time()
			regProcess = subprocess.call(regRun)
			endReg = time.time()	
			line = (str(numHash)+","+str(batchSize)+","+str(endGpu-startGpu)+
				","+str(endReg-startReg))
			f.write(line+"\n")
			print "Update: Wrote Hash: "+str(numHash)+" batchSize "+str(batchSize)
			gpuOpen = open("data/reg/result/gpu.txt","r+")
			gpuBloom = gpuOpen.readline()
			regOpen = open("data/reg/result/reg.txt","r+")
			regBloom = regOpen.readline()
			if(gpuBloom == regBloom):
				print "Correct"
			else:
				print "Incorrect"
			subprocess.call(["rm","-rf","data/reg/result/*"])
		f.close()				

if __name__=="__main__":
	hashStart = 1;
	hashEnd = 11
	hashInc = 2
	batchStart = 10000
	batchEnd = 60000
	batchInc= 10000
	parser = optparse.OptionParser()
	parser.add_option("--hashStart")
	parser.add_option("--hashEnd")
	parser.add_option("--hashInc")
	parser.add_option("--batchStart")
	parser.add_option("--batchEnd")
	parser.add_option("--batchInc")	
	(options,args) = parser.parse_args()
	if(options.hashStart is not None):
		hashStart = int(options.hashStart)
	if(options.hashEnd is not None):
		hashEnd = int(options.hashEnd)
	if(options.hashInc is not None):
		hashInc = int(options.hashInc)	
	if(options.batchStart is not None):
		batchStart = int(options.batchStart)
	if(options.batchEnd is not None):
		batchEnd = int(options.batchEnd)
	if(options.batchInc is not None):
		batchInc = int(options.batchInc)	
	main(hashStart = hashStart,hashEnd=hashEnd,hashInc=hashInc,
		batchStart = batchStart,batchEnd = batchEnd,batchInc = batchInc)
		
