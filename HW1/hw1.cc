#include<cstdio>
#include<stdlib.h>
#include<mpi.h>
#include<iostream>



void OddEvenSort(float arr[], int n) {
	bool isSorted = false;
	while (!isSorted) {
		isSorted = true;

		// even sort
		for (int i = 1; i <= n-2 ; i+=2){
			if(arr[i] > arr[i+1]){
				std::swap(arr[i], arr[i+1]);
				isSorted = false;
			}
		}

		for (int i=0; i <= n-2; i+=2){
			if(arr[i] > arr[i+1]){
				std::swap(arr[i], arr[i+1]);
				isSorted = false;
			}
		}
	}
	return;
}



int global_index(int rank, int k_idx, int size) {
	return rank*2 + k_idx*size*2;
}

// even_sort
bool EvenSort(float arr[], int k, int rank, int size, int n) {

	bool change = false;
	for (int i = 0; i < k; i++) {

		int idx = i*3;
		int global_idx = global_index(rank, i, size);
		int global_set = i * size + rank;

		if ( (global_idx + 1) <= (n-1)   &&  arr[idx] > arr[idx+1]) {
			std::swap(arr[idx], arr[idx+1]);
			change = true;
		}


		if (global_set > 0) {
			MPI_Send(&arr[idx], 1, MPI_FLOAT, (global_set-1)%size, i, MPI_COMM_WORLD);
		}

	}

	return change;
}


// odd_sort
bool OddSort(float arr[], int k, int rank, int size, int n) {
	static int global_set_boundary = (size*k)-1;
	bool change = false;

	for (int i = 0; i < k; i++) {
		int idx = i*3;
		int global_idx = global_index(rank, i, size);
		int global_set = i * size + rank;
		int recv_tag = i;
		if (global_set < global_set_boundary) {
			if(rank == size-1){
				recv_tag += 1; 
			}
			MPI_Recv(&arr[idx+2], 1, MPI_FLOAT, (global_set+1)%size, recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}


		if ( (global_idx +2) <= (n-1) && arr[idx+1] > arr[idx+2]) {
			std::swap(arr[idx+1], arr[idx+2]);
			change = true;
		}
	}
	return change;
}

int main(int argc, char** argv) {
	MPI_Init(&argc,&argv);
	int rank, size, size_;
	int n = atoi(argv[1]);
	char *file = argv[3];	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	size_ = size - 1;
	MPI_File f, out;
	MPI_Status status;
	MPI_Request request;
	// int phase[1] = {0}; // 0: even , 1: odd, 2: terminate
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out);
	// printf("argv[2] = %s \n",argv[2]);
	
	// need n/2 
	int needed_proc = n/2;
	int k = (needed_proc/size)+1 ; 
	if( k < 1){
		k = 1;		
	}

	


	if(size == 1){
		float data[n];
		for(int c = 0; c < n; c++){
			data[c] = -9;
			// printf("rank %d  c = %d got float: %f\n", rank,c, data[c]);
		}

		MPI_File_read_at(f, sizeof(float) * rank, &data, n, MPI_FLOAT, MPI_STATUS_IGNORE);

		OddEvenSort(data, n);
		for(int c = 0; c < n; c++){
			printf("%f\t", data[c]);
		}
		
		MPI_File_write_at(out, sizeof(float) * rank, &data, n, MPI_FLOAT, MPI_STATUS_IGNORE);
	}else{

		if(rank == 0){
			// recv isSorted
		}


		float data[3*k];
		for(int c = 0; c < 3*k; c++){
			data[c] = -9;
			// printf("rank %d  c = %d got float: %f\n", rank,c, data[c]);
		}
		for (int i = 0 ; i < k; i++){
			int global_idx = global_index(rank, i, size);
			MPI_File_read_at(f, sizeof(float) * (global_idx), &data[i*3], 3, MPI_FLOAT, MPI_STATUS_IGNORE);
		}
		// printf("rank = %d size = %d\n", rank, size);
		// MPI_File_write_at(f, ...
		MPI_Barrier (MPI_COMM_WORLD);

		EvenSort(data, k, rank, size, n);
		MPI_Barrier (MPI_COMM_WORLD);
		OddSort(data, k, rank, size, n);
		MPI_Barrier (MPI_COMM_WORLD);
		for(int c = 0; c < k; c++){
			// if((c*3)+2 > (3*k-1))
			// 	break;
			printf("rank %d  k  =  %d got float: %5.2f\t%5.2f\t%5.2f\n", rank, c, data[c*3],data[(c*3)+1],data[(c*3)+2]);

		}

		for (int i = 0 ; i < k; i++){
			int global_idx = global_index(rank, i, size);
			// MPI_File_read_at(f, sizeof(float) * (global_idx), &data[i*3], 3, MPI_FLOAT, MPI_STATUS_IGNORE);
			MPI_File_write_at(out, sizeof(float) * (global_idx), &data[i*3], 2, MPI_FLOAT,MPI_STATUS_IGNORE);

		}
	}

	


	int root = 0;
	// MPI_Barrier (MPI_COMM_WORLD);
	// if(rank  == 0){

	// }
	

	

	MPI_Finalize();
}
