import subprocess,sys
import time

def main(args):	
	#Firstly, vary the number of hash functions.
	for numHash in range(1,10000,10):
		f = open("data/pbf/time/numHash_"+str(numHash),"w+")
		f.write("#Hash,BatchSize(Bytes),PBFTime,RegTime\n")	
		#Now, vary size (in bytes) of the batches being inserted. 
		for batchSize in range(10000000,1000000000,10000000):
			#Remove any data that already exists in the directory.
			subprocess.call(["rm","-rf",".txt"])
			#Generate the batch data.
			subprocess.call(["bin/pbfmain.exe","--batchSize",str(batchSize),
				"--generate","--silent"])	
			#Setup the basic arguments.
			basicArgs = ["--hashes",str(numHash),"--prob",".02","--batchSize",
				str(batchSize),"--silent","--size","100000000"]
			pbfArgs = ["bin/pbfmain.exe"]
			pbfRun = []
			pbfRun[len(pbfRun):] = pbfArgs
			pbfRun[len(pbfRun):] = basicArgs
			#Add one more argument for the output of the PBF.
			#So, the stats of each PBF can be compared.	
			pbfRun[len(pbfRun):] = ["--pbfOutput",
				"data/pbf/pbfResult/gpu_"+str(numHash)+"_"+str(batchSize)+".txt"]		
			startPbf = time.time()		
			pbfProcess = subprocess.call(pbfRun)
			endPbf = time.time()
			regArgs = ["bin/pbfReg.exe"]
			regRun = []
			regRun[len(regRun):] = regArgs
			regRun[len(regRun):] = basicArgs
			regRun[len(regRun):] = ["--pbfOutput",
				"data/pbf/pbfResult/reg_"+str(numHash)+"_"+str(batchSize)+".txt"]
			startReg = time.time()
			regProcess = subprocess.call(regRun)
			endReg = time.time()
			line = (str(numHash)+","+str(batchSize)+","+str(endPbf-startPbf)+","+
				str(endReg-startReg))
			f.write(line+"\n")
			print "Update: Wrote hash: "+str(numHash)+" batchSize "+str(batchSize)
		f.close()
if __name__=="__main__":
	main(sys.argv[1:len(sys.argv)])
