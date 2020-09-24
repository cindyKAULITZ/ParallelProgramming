#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
  
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	double start, end;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long r_curr = 0;
	int rank, size, batch, proc = 0;
	MPI_Request request;
	MPI_Status status;
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	start = MPI_Wtime();
	batch = r/size;
	proc += 1;
	if(proc>=size) proc = 1;
	for (unsigned long long x = r_curr; (x < r) || ( x-r_curr < batch); x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
		pixels %= k;
	}
	if(r_curr+batch < r){
		r_curr += batch;
	}
	  
	if (rank == 0){
		// if(pixels != 0 ){
			printf("%llu\n", (4 * pixels) % k);
		// } 
	}else{
		MPI_Isend(
					&r_curr, 1, MPI_UNSIGNED_LONG_LONG,
					proc, 
					1, 
					MPI_COMM_WORLD, &request);
					
        MPI_Wait(&request, &status); 
	}
	end = MPI_Wtime();
	MPI_Finalize();
}
