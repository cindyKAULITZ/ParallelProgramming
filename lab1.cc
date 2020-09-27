#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
unsigned long long compute_single(unsigned long long x, unsigned long long r){
	unsigned long long y = ceil(sqrtl(r*r - (x)*(x)));
	// x += 1;
	printf("x = %f\n", x);
	printf("y = %f\n", y);
	return y; 
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	double start, end;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long pixels_single = 0;
	unsigned long long r_curr = 0;
	int rank, size, batch, proc = 0;

	MPI_Request request;
	MPI_Status status;
	// MPI_Comm &comm;

    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	printf("in proc %d\n", rank);
	printf("r_curr = %d\n", r_curr);  
	printf("pixels_single = %f\n", pixels_single);  
	pixels_single = compute_single(r_curr, r);
	r_curr+=1;

	start = MPI_Wtime();
	batch = r/size;
	proc += 1;
	if(proc>=size) proc = 1;
	foru(r_curr < r){
			MPI_Isend(
						&r_curr, 1, MPI_UNSIGNED_LONG_LONG,
						proc, 
						1, 
						MPI_COMM_WORLD, &request);
	}
	// split sum to one task via MPI_Reduce()
	// for (unsigned long long x = r_curr; (x < r) || ( x-r_curr < batch); x++) {
		
	// 	pixels += y;
	// 	pixels %= k;
	// }
	// MPI_Reduce(&pixels_single, &pixels, 0,MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0){
		// for unsigned long long x = r_curr; r_curr < r; r_curr++){
		
		// }
		// MPI_Wait(&request, &status); 
		printf("pixels = %f\n", pixels);
		pixels%=k;	
		printf("%llu\n", (4 * pixels) % k);

	}else {
	}
	end = MPI_Wtime();
	MPI_Finalize();
}
