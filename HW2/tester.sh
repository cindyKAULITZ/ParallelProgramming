# hwa2 test
make 
srun -n1 -c4 ./hw2a out.png 10000 -2 2 -2 2 800 800
hw2-diff out.png pthread_test.png