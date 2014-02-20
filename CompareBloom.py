import subprocess,sys
import time


def main(argv):
	#Vary the number of hash functions...
	for numHash in range(1,10000,10):
		f=open("data/reg/time/numHash_"+str(numHash),"w+")
		f.write("#Hash,BatchSize(Bytes),GPUTime,RegTime\n")
		#Now, vary size (in bytes) of the batches being inserted.
		for batchSize in range(10000000,1000000000,10000000):
			#Remove any data that already exists in the directory.
			subprocess.call(["rm","-rf",".txt"])
			#Generate the batch data.	
			subprocess.call(["bin/main.exe","--batchSize",str(batchSize),
				"--generate","--silent"])
			#Setup the basic arguments.
			basicArgs = ["--hashes",str(numHash),"--batchSize",str(batchSize),
				"--silent","--size","100000000"]
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
	main(sys.argv[0:len(sys.argv)])

