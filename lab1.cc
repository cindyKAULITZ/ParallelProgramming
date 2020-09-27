#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>


unsigned long long compute_single(unsigned long long x, unsigned long long r_square){
	unsigned long long y = ceil(sqrtl(r_square - (x*x)));
	return y; 
}

  
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
    
    // pre-calculation
    r_squre = r*r;
    // elements_per_process =  floorl(r + np - 1) / np; 

    // np -> number of processes 
    // pid -> process id 

    MPI_Status status; 
    // Creation of parallel processes 
    MPI_Init(&argc, &argv); 
    // find out process ID, 
    // and how many processes were started 
    MPI_Comm_rank(MPI_COMM_WORLD, &pid); 
    MPI_Comm_size(MPI_COMM_WORLD, &np); 

	// master process 
    if (pid == 0) { 
        int i; 
        
        // r_squre = r*r;
        elements_per_process =  (r + np - 1) / np; 


        // check if more than 1 processes are run 
        if (np > 1) { 
            // distributes the necessary information for each processor
            for (i = 1; i < np ; ++i) { 
                sub_start = i * elements_per_process; 
  
                MPI_Send(&elements_per_process, 
                         1, 
						 MPI_UNSIGNED_LONG_LONG, i, 0, 
                         MPI_COMM_WORLD); 

                MPI_Send(&sub_start, 
                         1, 
                         MPI_UNSIGNED_LONG_LONG, i, 1, 
                         MPI_COMM_WORLD); 
            } 
        } 
  
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
            int sender = status.MPI_SOURCE; 
            sum += tmp; 
            sum %= k;
        } 

        // prints the final sum of pixels 
		printf("%llu\n", (sum << 2) % k);
    } 
    // slave processes 
    else { 
        MPI_Recv(&elements_per_process, 
                 1, 
				 MPI_UNSIGNED_LONG_LONG, 0, 0, 
                 MPI_COMM_WORLD, 
                 &status); 

		MPI_Recv(&sub_start, 
                 1, 
				 MPI_UNSIGNED_LONG_LONG, 0, 1, 
                 MPI_COMM_WORLD, 
                 &status); 

        // calculates its partial sum of pixel
        unsigned long long partial_sum = 0; 
		unsigned long long sub_end = sub_start + elements_per_process;
		if (pid == (np-1)) { sub_end = r; }
        
		// subtask
		for (unsigned long long i = sub_start; i < sub_end; ++i) {
            partial_sum += compute_single(i, r_squre); 
        }
  
        // sends the partial sum to the root process 
        MPI_Send(&partial_sum, 1, MPI_UNSIGNED_LONG_LONG, 
                 0, 0, MPI_COMM_WORLD); 
    } 
  
    // cleans up all MPI state before exit of process 
    MPI_Finalize(); 
  
    return 0; 
}


// nproc=12
// r=4294967295
// k=1099511627775
// answer=576603832986
// timelimit=20
