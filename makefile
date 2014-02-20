NVCC = /usr/local/cuda/bin/nvcc

make gpu: data/ bin/ bin/gpuMain.exe

make reg: data/ bin/ bin/main.exe 

make pbf: data/ bin/ bin/pbfmain.exe

make pbfReg: data/ bin/ bin/pbfReg.exe

data/:
	mkdir data
	mkdir data/pbf
	mkdir data/pbf/pbfResult
	mkdir data/pbf/time
	mkdir data/reg
	mkdir data/reg/result
	mkdir data/reg/time
bin/: 
	mkdir bin

bin/main.exe: bin/main.o bin/ParseArgs.o bin/RandomGenerator.o bin/Hash.o bin/ParseData.o 
	g++ bin/main.o bin/ParseArgs.o bin/RandomGenerator.o bin/Hash.o bin/ParseData.o -o bin/main.exe

bin/main.o: regularBloom/main.cpp 
	g++ -c regularBloom/main.cpp 
	mv main.o bin/

bin/ParseArgs.o: misc/ParseArgs.cpp ParseArgs.h
	g++ -c misc/ParseArgs.cpp
	mv ParseArgs.o bin/

bin/RandomGenerator.o: RandomGenerator.h misc/RandomGenerator.cpp
	g++ -c misc/RandomGenerator.cpp
	mv RandomGenerator.o bin/

bin/ParseData.o: ParseData.h misc/ParseData.cpp
	g++ -c misc/ParseData.cpp
	mv ParseData.o bin/

bin/Hash.o: regularBloom/Hash.h regularBloom/Hash.cpp
	g++ -c regularBloom/Hash.cpp
	mv Hash.o bin/

bin/gpuMain.exe: bin/RandomGenerator.o bin/ParseArgs.o bin/ParseData.o gpguBloom/gpumain.cpp bin/bloom.o gpguBloom/Bloom.h
	$(NVCC) -arch=sm_11 -o bin/gpuMain.exe gpguBloom/gpumain.cpp  bin/ParseArgs.o bin/RandomGenerator.o bin/ParseData.o bin/bloom.o

bin/pbfmain.exe: bin/RandomGenerator.o bin/ParseArgs.o bin/ParseData.o gpguBloom/pbfmain.cpp bin/bloom.o gpguBloom/Bloom.h bin/pbfstats.o
	$(NVCC) -arch=sm_11 -o bin/pbfmain.exe gpguBloom/pbfmain.cpp  bin/ParseArgs.o bin/RandomGenerator.o bin/ParseData.o bin/bloom.o bin/pbfstats.o

bin/pbfReg.exe: bin/RandomGenerator.o bin/ParseArgs.o bin/ParseData.o regularBloom/pbfReg.cpp bin/Hash.o bin/pbfstats.o
	g++ regularBloom/pbfReg.cpp bin/ParseArgs.o bin/RandomGenerator.o bin/Hash.o bin/ParseData.o bin/pbfstats.o -o bin/pbfReg.exe

bin/pbfstats.o: PBFStats.h misc/PBFStats.cpp
	g++ -c misc/PBFStats.cpp
	mv PBFStats.o bin/pbfstats.o	

bin/bloom.o: gpguBloom/bloom.cu
	$(NVCC) -arch=sm_11 -c gpguBloom/bloom.cu 
	mv bloom.o bin/
	
clean:
	rm -rf bin
	rm -rf data
	rm *.txt
	
run: 
	./bin/main.exe	
