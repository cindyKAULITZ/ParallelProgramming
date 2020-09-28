#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
// unsigned long long compute_single(unsigned long long x, unsigned long long r){
// 	unsigned long long y = ceil(sqrtl(r*r - (x)*(x)));
// 	// x += 1;
// 	// printf("x = %f\n", x);
// 	// printf("y = %f\n", y);
// 	return y; 
// }

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
	// unsigned long long r_curr = 0;
	unsigned long long r_square, batch, index = 0;
	int rank, size;
	unsigned long long send_id = 1; 
	unsigned long long w = 0;

	// MPI_Request request;
	MPI_Status status;
	// MPI_Comm &comm;

    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	// start = MPI_Wtime();
	// batch = (r + size - 1) / size;
	// for(int i = 1 ; i < size-1; i++){
	// 	index[i] = (i*batch);
	// }
	
	
	if (rank == 0){
		unsigned long long r_square = r*r;
		w = ceil(sqrtl(r_square - (r-1)*(r-1)));
		printf("w = %llu\n", w);
		unsigned long long sum = 0;
		int c = 0;
		for(unsigned long long x_start = w; x_start <= (r-2); x_start++){
			sum += ceil(sqrtl(r_square-(x_start*x_start)));
			c++;

		}
		sum += w*(r+1);
		printf("sum = %llu\n", sum );
		printf("c = %d\n", c);
		// printf("%sum ans = llu\n", (4 * sum) % k);
		// unsigned long long new_r = ((r-2)-w)+1;
		// printf("new_R = %llu\n", new_r);

		batch = ((r-1)/(size))+1;
		printf("batch = %llu\n", batch);
		for(send_id = 1; send_id < (size); send_id++){
			// MPI_Isend(buffer,count,type,dest,tag,comm,request)
			index =  (send_id)*batch;
			// printf("r_curr = %d\n", r_curr);			
			// printf("rbatch = %d\n", batch);			
			// printf("index = %d\n", index);			
			MPI_Send(
					&index, 1, MPI_UNSIGNED_LONG_LONG,
					send_id, 
					1, 
					MPI_COMM_WORLD);
			MPI_Send(
					&r_square, 1, MPI_UNSIGNED_LONG_LONG,
					send_id, 
					2, 
					MPI_COMM_WORLD);
			MPI_Send(
					&batch, 1, MPI_UNSIGNED_LONG_LONG,
					send_id, 
					3, 
					MPI_COMM_WORLD);
			MPI_Send(
					&w, 1, MPI_UNSIGNED_LONG_LONG,
					send_id, 
					4, 
					MPI_COMM_WORLD);
			// MPI_Isend(
			// 		&send_id, 1, MPI_UNSIGNED_LONG_LONG,
			// 		send_id, 
			// 		1, 
			// 		MPI_COMM_WORLD, &request);

		}
		// MPI_Wait(&request, &status);
		
		unsigned long long sub_end = batch;
		if (sub_end > r) { sub_end = r; }
		for(unsigned long long ele_count = 0; ele_count < sub_end; ele_count++){
			// printf("r_curr = %d\n", r_curr);
			// unsigned long long y = ;
			pixels += ceil(sqrtl(r_square - ele_count*ele_count));
			// pixels %= k;
		}
		// pixels += pixels_single;



		unsigned long long tmp;
        for (int recv_id = 1; recv_id < size; recv_id++) { 
            MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG_LONG, 
                     MPI_ANY_SOURCE, 1, 
                     MPI_COMM_WORLD, 
                     &status); 
            // int sender = status.MPI_SOURCE; 
            pixels += tmp;
			// pixels %= k; 
			// printf("pixels = %llu\n", pixels);
        } 

		// printf("pixels = %llu\n", pixels);
		// pixels += w*(r+1);
		// printf("w = %llu\n", w);	
		printf("pixels = %llu\n", pixels);

		printf("%llu\n", (4 * pixels) % k);

	}else {

		MPI_Recv(&index, 1, MPI_UNSIGNED_LONG_LONG, 
                     0, 1, 
                     MPI_COMM_WORLD, 
                     &status); 
		MPI_Recv(&r_square, 1, MPI_UNSIGNED_LONG_LONG, 
                     0, 2, 
                     MPI_COMM_WORLD, 
                     &status); 
		MPI_Recv(&batch, 1, MPI_UNSIGNED_LONG_LONG, 
                     0, 3, 
                     MPI_COMM_WORLD, 
                     &status); 
		MPI_Recv(&w, 1, MPI_UNSIGNED_LONG_LONG, 
                     0, 4, 
                     MPI_COMM_WORLD, 
                     &status); 

	
		unsigned long long sub_end = index+batch;
		if (sub_end >r) { sub_end = r; }
		for(unsigned long long ele_count = index; ele_count < sub_end; ele_count++){
			// printf("r_curr = %d\n", r_curr);
			// unsigned long long y = ;
			pixels_single += ceil(sqrtl(r_square - ele_count*ele_count));
			// pixels_single %= k;
		}
		// r_curr+=batch;
		MPI_Send(
					&pixels_single, 1, MPI_UNSIGNED_LONG_LONG,
					0, 
					1, 
					MPI_COMM_WORLD);
		// MPI_Isend(
		// 			&r_curr, 1, MPI_UNSIGNED_LONG_LONG,
		// 			0, 
		// 			1, 
		// 			MPI_COMM_WORLD, &request);
	}


	// MPI_Reduce(&pixels_single, &pixels, 0,MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	
	// end = MPI_Wtime();
	MPI_Finalize();
}
