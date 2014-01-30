NVCC = /usr/local/cuda/bin/nvcc

make gpu: bin/ bin/gpuMain.exe

make reg: bin/ bin/main.exe 

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
	$(NVCC) -o bin/gpuMain.exe gpguBloom/gpumain.cpp  bin/ParseArgs.o bin/RandomGenerator.o bin/ParseData.o bin/bloom.o -arch sm_20 
	 
bin/bloom.o: gpguBloom/bloom.cu
	$(NVCC) -c gpguBloom/bloom.cu -arch sm_20
	mv bloom.o bin/
	
clean:
	rm bin/*
	rm *.txt
	
run: 
	./bin/main.exe	
