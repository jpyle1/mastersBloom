#!/bin/bash

#Clean up the already generated files.
rm -rf *.txt

#Arguments to pass in the 
PARAMETERS=" "
#Command used to launch the cuda program.
COMMAND=" "

#Get the arguments the user specified.
while getopts $ARGV "s:h:b:n:p:t:w": CURRENT_OPTION
do 
	case $CURRENT_OPTION in
		s)
			PARAMETERS=$PARAMETERS:" --size $OPTARG "
		;;
		h)
			PARAMETERS=$PARAMETERS:" --hashes $OPTARG "
		;;
		b)
			PARAMETERS=$PARAMETERS:" --batchSize $OPTARG "
		;;
		n)
			PARAMETERS=$PARAMETERS:" --numBatches $OPTARG "
		;;
		t)
			PARAMETERS=$PARAMETERS:" -tb $OPTARG "
		;;
		w)
			PARAMETERS=$PARAMETERS:" --falseBatches $OPTARG "
		;;

		p)
			COMMAND=" $OPTARG "	
		;;
		
	
	esac
done

echo "Generating data"
GEN=" --generate --silent "
time ./bin/main.exe $GEN:$PARAMETERS

#Insert items into the gpu bloom
echo "Starting gpu bloom"
GPUCOMMAND=" -f gpuOut.txt  ":$PARAMETERS
time $COMMAND ./bin/gpuMain.exe $GPUCOMMAND

#Insert items into the bloom filters.
echo "Starting regular"
REGCOMMAND=" -f regOut.txt ":$PARAMETERS
time ./bin/main.exe $REGCOMMAND



#Compare the two bloom filters and make sure that the results match.
gpuBloom=$(<gpuOut.txt)
regBloom=$(<regOut.txt)

echo "===="
if [ $gpuBloom == $regBloom ]
then
	echo "Correct output"
else
	echo "Incorrect output"
fi

