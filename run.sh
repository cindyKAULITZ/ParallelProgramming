#!/bin/bash
# filelist = "10000.txt 20000.txt 30000.txt 40000.txt 50000.txt"
# modelist = "0 1"

for file in 10000.txt 20000.txt 30000.txt 40000.txt 50000.txt
do 
    echo 1 $file
    srun -N1 -n1 -c1 --gres=gpu:1 ./bin/main findbest ./datasets/S1_Dataset/$file 1

done

for file in 10000.txt 20000.txt 30000.txt 40000.txt 50000.txt
do 
    echo 0 $file
    ./bin/main findbest ./datasets/S1_Dataset/$file 0

done
