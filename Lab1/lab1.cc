#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
// static unsigned long long compute_single(unsigned long long x, unsigned long long r_square){
// 	return ceil(sqrtl(r_square - (x)*(x))); 
// }

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	const unsigned long long r = atoll(argv[1]);
	const unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long pixels_single = 0;
	
	int rank, size;
	double start, end;

	MPI_Request request;
	MPI_Status status;
	const unsigned long long r_square = r*r;

	// start parallel computation
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	// start = MPI_Wtime();
	const unsigned long long batch = ((r-1)/(size))+1;
	
	//caculate partial sum
	unsigned long long sub_start = (unsigned long long)rank * batch;
	unsigned long long sub_end = sub_start + batch;
	if (sub_end > r) { sub_end = r;}
	for(unsigned long long ele_count = sub_start; ele_count < sub_end; ele_count++){
		pixels_single += ceil(sqrtl(r_square - (ele_count)*(ele_count)));
	}
	pixels_single %= k;

	// Reduce all partial sum to processor 0 to pixels
	MPI_Reduce(&pixels_single, &pixels, 1,MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	// end = MPI_Wtime();

	if(rank == 0){
		printf("%llu\n", (pixels*4) % k);
		// printf("%f\n", end-start);
	}
	MPI_Finalize();
}
