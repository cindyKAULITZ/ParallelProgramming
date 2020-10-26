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
int EvenSort(float arr[], int set_pos[], int global_index_arr[], int k, int rank, int size, int n) {

	static bool init = false;
	// static int global_set_boundary = n/2 - 1;

	int change = 0;
	for (int i = 0; i < k; i++) {
		int idx = set_pos[i];
		int global_idx = global_index_arr[i];
		int global_set = i * size + rank;

		if (init) {
			if (global_set > 0) {
						// if (global_set > 0 && global_set < global_set_boundary ) {

				MPI_Recv(&arr[idx], 1, MPI_FLOAT, (global_set-1)%size, global_set-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}


		if ( (global_idx + 1) <= (n-1)   &&  arr[idx] > arr[idx+1]) {
			std::swap(arr[idx], arr[idx+1]);
			change = 1;
		}

		if (global_set > 0) {
						// if (global_set > 0 && global_set < global_set_boundary ) {

			MPI_Send(&arr[idx], 1, MPI_FLOAT, (global_set-1)%size, (global_set-1), MPI_COMM_WORLD);
		}

	}

	if (!init) {
		init = true;
	}
	return change;
}


// odd_sort
int OddSort(float arr[], int set_pos[], int global_index_arr[], int k, int rank, int size, int n) {
	static int global_set_boundary = (size*k)-1;
	// static int global_set_boundary = n/2 -1;

	int change = 0;

	for (int i = 0; i < k; i++) {
		int idx = set_pos[i];
		int global_idx = global_index_arr[i];
		int global_set = i * size + rank;
		if (global_set < global_set_boundary) {
			MPI_Recv(&arr[idx+2], 1, MPI_FLOAT, (global_set+1)%size, global_set, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}


		if ( (global_idx +2) <= (n-1) && arr[idx+1] > arr[idx+2]) {
			std::swap(arr[idx+1], arr[idx+2]);
			change = 1;
		}

		if (global_set < global_set_boundary) {
			// MPI_Request request;
			MPI_Send(&arr[idx+2], 1, MPI_FLOAT, (global_set+1)%size, (global_set), MPI_COMM_WORLD);
		}
	}
	return change;
}

int main(int argc, char** argv) {
	MPI_Init(&argc,&argv);
	int rank, size;
	int n = atoll(argv[1]);
	char *file = argv[3];	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_File f, out;
	MPI_Status status;
	MPI_Request request;
	int change_odd = 1, change_even = 1, change = 0, isSorted = 1;
	// int phase[1] = {0}; // 0: even , 1: odd, 2: terminate
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out);
	// printf("argv[2] = %s \n",argv[2]);
	
	// need n/2 
	int needed_proc = n/2;
	int k = (needed_proc/size)+1 ; 


	if(size == 1){
		float data[n];
		MPI_File_read_at(f, sizeof(float) * rank, &data, n, MPI_FLOAT, MPI_STATUS_IGNORE);
		OddEvenSort(data, n);		
		MPI_File_write_at(out, sizeof(float)*rank, &data, n, MPI_FLOAT, MPI_STATUS_IGNORE);
	} else {
		int set_pos[k];
		int global_index_arr[k];
		for(int i = 0; i < k; i++){
			set_pos[i] = i*3 ;
			global_index_arr[i] = global_index(rank, i, size);
		}
		
		float data[3*k];
		// for(int c = 0; c < 3*k; c++){
		// 	data[c] = -9;
		// 	// printf("rank %d  c = %d got float: %f\n", rank,c, data[c]);
		// }
		for (int i = 0 ; i < k; i++){
			int global_idx = global_index_arr[i];
			MPI_File_read_at(f, sizeof(float) * (global_idx), &data[set_pos[i]], 3, MPI_FLOAT, MPI_STATUS_IGNORE);
		}
		MPI_Barrier(MPI_COMM_WORLD);

		while(isSorted != 0){
			// printf("rank %d isSorted %d\n", rank, isSorted);
			change_even = EvenSort(data, set_pos, global_index_arr, k, rank, size, n);
			MPI_Barrier (MPI_COMM_WORLD);
			change_odd = OddSort(data,set_pos, global_index_arr, k, rank, size, n);
			MPI_Barrier (MPI_COMM_WORLD);
			change = change_even + change_odd;
			MPI_Barrier (MPI_COMM_WORLD);

			MPI_Allreduce(&change, &isSorted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			// printf("rank %d isSorted %d(after)\n", rank, isSorted);
			MPI_Barrier (MPI_COMM_WORLD);
			// MPI_Send(&change_odd, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
		}
		
		// for(int c = 0; c < k; c++){
		// 		printf("rank %d  k  =  %d got float: %5.2f\t%5.2f\t%5.2f\n", rank, c, data[c*3],data[(c*3)+1],data[(c*3)+2]);
		// }
		
		for (int i = 0 ; i < k; i++){
			int global_idx = global_index_arr[i];
			if(global_idx < n){
				MPI_File_write_at(out, sizeof(float) * (global_idx), &data[set_pos[i]], 1, MPI_FLOAT,MPI_STATUS_IGNORE);
			}
			if(global_idx+1 < n){
				MPI_File_write_at(out, (sizeof(float) * (global_idx+1)), &data[set_pos[i]+1], 1, MPI_FLOAT,MPI_STATUS_IGNORE);
			}
		}
	}
	
	MPI_Finalize();
}
