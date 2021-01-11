#!/bin/bash
# filelist = "10000.txt 20000.txt 30000.txt 40000.txt 50000.txt"
# modelist = "0 1"

for file in 50000.txt
do
    for b_s in 10000
    do
        echo ----------------------------------------------------
        echo mode : gpu, k = $1, b_s : $b_s train : $file
        # srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,inst_integer,gld_throughput,gst_throughput ./bin/main run ./datasets/S1_Dataset/$file ./datasets/S1_Dataset/10000.txt $1 1 $b_s
        # srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./bin/main run ./datasets/S1_Dataset/$file ./datasets/S1_Dataset/10000.txt $1 1 $b_s
        srun -N1 -n1 -c1 --gres=gpu:1 ./bin/main run ./datasets/S1_Dataset/$file ./datasets/S1_Dataset/10000.txt $1 1 $b_s
    done
done
for file in 50000.txt
do
    for b_s in 10000
    do
        echo ----------------------------------------------------
        echo mode : gpu, k = $1, b_s : $b_s train : $file
        # srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,inst_integer,gld_throughput,gst_throughput ./bin/main run ./datasets/S1_Dataset/$file ./datasets/S1_Dataset/10000.txt $1 1 $b_s
        # srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./bin/main run ./datasets/S1_Dataset/$file ./datasets/S1_Dataset/10000.txt $1 1 $b_s
        srun -N1 -n1 -c1 --gres=gpu:1 nvprof ./bin/main run ./datasets/S1_Dataset/$file ./datasets/S1_Dataset/10000.txt $1 1 $b_s
    done
done
# echo ----------------------------------------------------
# echo ----------------------------------------------------
# for file in 10000.txt 20000.txt 30000.txt 40000.txt 50000.txt
# do 
#     echo ----------------------------------------------------
#     echo mode : seq, k = $1, b_s : $b_s, train : $file
#     ./bin/main run ./datasets/S1_Dataset/$file ./datasets/S1_Dataset/10000.txt $1 0 $b_s

# done
# echo ------------------------prof----------------------------
# for file in 10000.txt 20000.txt 30000.txt 40000.txt 50000.txt
# do
#     for b_s in 10000
#     do
#         echo ----------------------------------------------------
#         echo mode : gpu, k = $1, b_s : $b_s train : $file
#         # srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,inst_integer,gld_throughput,gst_throughput ./bin/main run ./datasets/S1_Dataset/$file ./datasets/S1_Dataset/10000.txt $1 1 $b_s
#         # srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput ./bin/main run ./datasets/S1_Dataset/$file ./datasets/S1_Dataset/10000.txt $1 1 $b_s
#         srun -N1 -n1 -c1 --gres=gpu:1 nvprof ./bin/main run ./datasets/S1_Dataset/$file ./datasets/S1_Dataset/10000.txt $1 1 $b_s
#     done
# done
