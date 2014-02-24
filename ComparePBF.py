import subprocess,sys,time,optparse

def main(hashStart,hashEnd,hashInc,batchStart,batchEnd,batchInc):	
	#Firstly, vary the number of hash functions.
	for numHash in range(hashStart,hashEnd+hashInc,hashInc):
		f = open("data/pbf/time/numHash_"+str(numHash),"w+")
		f.write("#Hash,BatchSize(Bytes),PBFTime,RegTime\n")	
		#Now, vary size (in bytes) of the batches being inserted. 
		for batchSize in range(batchStart,batchEnd+batchInc,batchInc):
			#Remove any data that already exists in the directory.
			subprocess.call(["rm","-rf",".txt"])
			#Generate the batch data.
			subprocess.call(["bin/pbfmain.exe","--batchSize",str(batchSize),
				"--generate","--silent","--trueBatches","1","--falseBatches","1",
				"--numBatches","1"])	
			#Setup the basic arguments.
			basicArgs = ["--hashes",str(numHash),"--prob",".02","--batchSize",
				str(batchSize),"--silent","--size","100000000",
				"--numTrueBatchInsertions","1","--trueBatches","1","--falseBatches",
				"1"]
			pbfArgs = ["bin/pbfmain.exe"]
			pbfRun = []
			pbfRun[len(pbfRun):] = pbfArgs
			pbfRun[len(pbfRun):] = basicArgs
			#Add one more argument for the output of the PBF.
			#So, the stats of each PBF can be compared.	
			#pbfRun[len(pbfRun):] = ["--pbfOutput",
			#	"data/pbf/pbfResult/gpu_"+str(numHash)+"_"+str(batchSize)+".txt"]		
			startPbf = time.time()		
			pbfProcess = subprocess.call(pbfRun)
			endPbf = time.time()
			regArgs = ["bin/pbfReg.exe"]
			regRun = []
			regRun[len(regRun):] = regArgs
			regRun[len(regRun):] = basicArgs
			#regRun[len(regRun):] = ["--pbfOutput",
			#	"data/pbf/pbfResult/reg_"+str(numHash)+"_"+str(batchSize)+".txt"]
			startReg = time.time()
			regProcess = subprocess.call(regRun)
			endReg = time.time()
			line = (str(numHash)+","+str(batchSize)+","+str(endPbf-startPbf)+","+
				str(endReg-startReg))
			f.write(line+"\n")
			print "Update: Wrote hash: "+str(numHash)+" batchSize "+str(batchSize)
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
