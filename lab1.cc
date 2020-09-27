#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>


unsigned long long compute_single(unsigned long long x, unsigned long long r_square){
	unsigned long long y = ceil(sqrtl(r_square - (x*x)));
	return y; 
}


unsigned long long  Reduce_sum;

  
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	// double start, end;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);

	int pid, np;
	unsigned long long elements_per_process, sub_start, r_squre;
    unsigned long long i; 
    

    // np -> number of processes 
    // pid -> process id 

    MPI_Status status; 
    // Creation of parallel processes 
    MPI_Init(&argc, &argv); 
    // find out process ID, 
    // and how many processes were started 
    MPI_Comm_rank(MPI_COMM_WORLD, &pid); 
    MPI_Comm_size(MPI_COMM_WORLD, &np); 

    // pre-calculation
    r_squre = r*r;
    elements_per_process =  floorl(r + np - 1) / np; 
    sub_start = pid * elements_per_process;
    unsigned long long sub_end = sub_start + elements_per_process;

	// master process 
    if (pid == 0) {         
        // check if more than 1 processes are run 
        // if (np > 1) { 
        //     // distributes the necessary information for each processor
        //     // for (i = 1; i < np ; ++i) { 
        //     //     sub_start += elements_per_process; 
        //     //     MPI_Send(&sub_start, 
        //     //              1, 
        //     //              MPI_UNSIGNED_LONG_LONG, i, 1, 
        //     //              MPI_COMM_WORLD); 
        //     // } 
        // } 
  
        // own calculation of master process 
        unsigned long long sum = 0; 
        for (i = 0; i < elements_per_process; ++i) {
			sum += compute_single(i, r_squre);
		}
        sum %= k;
  
        // collects partial sums from other processes 
        unsigned long long tmp; 
        for (i = 1; i < np; i++) { 
            MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG_LONG, 
                     MPI_ANY_SOURCE, 0, 
                     MPI_COMM_WORLD, 
                     &status); 
            // int sender = status.MPI_SOURCE; 
            sum += tmp; 
        } 

        // sum += Reduce_sum;
        // prints the final sum of pixels 
		printf("%llu\n", (sum << 2) % k);
    } 
    // slave processes 
    else { 
		// MPI_Recv(&sub_start, 
        //          1, 
		// 		 MPI_UNSIGNED_LONG_LONG, 0, 1, 
        //          MPI_COMM_WORLD, 
        //          &status); 

        // calculates its partial sum of pixel
        unsigned long long partial_sum = 0; 
		if (pid == (np-1)) { sub_end = r; }
        
		// ubtask
		for (i = sub_start; i < sub_end; ++i) {
            partial_sum += compute_single(i, r_squre); 
        }
        partial_sum %= k;
  
        // sends the partial sum to the root process 
        MPI_Send(&partial_sum, 
                 1, 
                 MPI_UNSIGNED_LONG_LONG, 0, 0, 
                 MPI_COMM_WORLD); 
        // MPI_Reduce(&partial_sum, &Reduce_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
        //            MPI_COMM_WORLD);
    } 
  
    // cleans up all MPI state before exit of process 
    MPI_Finalize();   
    return 0; 
}

