#include<cstdio>
#include<stdlib.h>
#include<mpi.h>
#include<iostream>
#include <omp.h>


void OddEvenSort(float arr[], int n) {
	bool isSorted = false;	
	while (!isSorted) {
		isSorted = true;

		// even sort
		#pragma omp parallel for
		for (int i = 1; i <= n-2 ; i+=2) {
			if(arr[i] > arr[i+1]){
				std::swap(arr[i], arr[i+1]);
				isSorted = false;
			}
		}

		#pragma omp parallel for
		for (int i = 0; i <= n-2; i+=2) {
			if(arr[i] > arr[i+1]){
				std::swap(arr[i], arr[i+1]);
				isSorted = false;
			}
		}
	}
	return;
}



unsigned long long global_index(unsigned long long rank, unsigned long long k_idx, unsigned long long size) {
	return rank*2 + k_idx*size*2;
}


// even_sort
int EvenSort(float arr[], unsigned long long set_pos[], unsigned long long global_index_arr[], unsigned long long k, unsigned long long rank, unsigned long long size, unsigned long long n) {
	static bool init = false;
	static unsigned long long global_set_mod_factor = 2 * size;

	int change = 0;
	#pragma omp parallel for
	for (unsigned long long i = 0; i < k; i++) {
		unsigned long long idx = set_pos[i];
		unsigned long long global_idx = global_index_arr[i];
		unsigned long long global_set = i * size + rank;

		if (init) {
			if (global_set > 0) {
				MPI_Recv(&arr[idx], 1, MPI_FLOAT, (global_set-1)%size, (global_set-1)%global_set_mod_factor, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}

		if ( global_idx + 1  <= (n-1)   &&  arr[idx] > arr[idx+1]) {
			std::swap(arr[idx], arr[idx+1]);
			change = 1;
		}

		if (global_set > 0) {
			MPI_Send(&arr[idx], 1, MPI_FLOAT, (global_set-1)%size, (global_set-1)%global_set_mod_factor, MPI_COMM_WORLD);
		}
	}

	if (!init) {
		init = true;
	}
	return change;
}


// odd_sort
int OddSort(float arr[], unsigned long long set_pos[], unsigned long long global_index_arr[], unsigned long long k, unsigned long long rank, unsigned long long size, unsigned long long n) {
	static unsigned long long global_set_boundary = (size*k)-1;
	static unsigned long long global_set_mod_factor = 2 * size;

	int change = 0;
	#pragma omp parallel for
	for (unsigned long long i = 0; i < k; i++) {
		unsigned long long idx = set_pos[i];
		unsigned long long global_idx = global_index_arr[i];
		unsigned long long global_set = i * size + rank;
		if (global_set < global_set_boundary) {
			MPI_Recv(&arr[idx+2], 1, MPI_FLOAT, (global_set+1)%size, global_set%global_set_mod_factor, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		if ( global_idx + 2 <= (n-1) && arr[idx+1] > arr[idx+2]) {
			std::swap(arr[idx+1], arr[idx+2]);
			change = 1;
		}

		if (global_set < global_set_boundary) {
			MPI_Send(&arr[idx+2], 1, MPI_FLOAT, (global_set+1)%size, global_set%global_set_mod_factor, MPI_COMM_WORLD);
		}
	}
	return change;
}


// even_sort
void GarbageRecv(unsigned long long k, unsigned long long rank, unsigned long long size) {
	static unsigned long long global_set_mod_factor = 2*size;

	#pragma omp parallel for
	for (unsigned long long i = 0; i < k; i++) {
		unsigned long long global_set = i * size + rank;
		float garbage;
		if (global_set > 0) {
			MPI_Recv(&garbage, 1, MPI_FLOAT, (global_set-1)%size, (global_set-1)%global_set_mod_factor, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
}

int main(int argc, char** argv) {
	MPI_Init(&argc,&argv);
	int rank, size;
	unsigned long long n = atoll(argv[1]);
	char *file = argv[3];	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_File f, out;
	int change_odd = 1, change_even = 1, change = 0, isSorted = 1;
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out);

	// need n/2 
	unsigned long long needed_proc = n/2;
	unsigned long long k = (needed_proc/size)+1;

	if(size == 1){
		float *data = (float*)malloc(sizeof(float)*n); 
		MPI_File_read_at(f, sizeof(float)*rank, data, n, MPI_FLOAT, MPI_STATUS_IGNORE);
		OddEvenSort(data, n);		
		MPI_File_write_at(out, sizeof(float)*rank, data, n, MPI_FLOAT, MPI_STATUS_IGNORE);
		free(data);
	} else {
		unsigned long long *set_pos = (unsigned long long*)malloc(sizeof(unsigned long long)*k);
		unsigned long long *global_index_arr = (unsigned long long*)malloc(sizeof(unsigned long long)*k);;
		#pragma omp parallel for
		for(unsigned long long i = 0; i < k; ++i){
			set_pos[i] = i*3 ;
			global_index_arr[i] = global_index(rank, i, size);
		}
		
		float *data = (float*)malloc(sizeof(float)*3*k); 
		// for(int c = 0; c < 3*k; c++){
		// 	data[c] = -0.99999999;
		// 	// printf("rank %d  c = %d got float: %f\n", rank,c, data[c]);
		// }

		#pragma omp parallel for
		for (unsigned long long i = 0 ; i < k; i++){
			unsigned long long global_idx = global_index_arr[i];
			MPI_File_read_at(f, sizeof(float)*(global_idx), &data[set_pos[i]], 3, MPI_FLOAT, MPI_STATUS_IGNORE);
		}

		// for(int c = 0; c < k; c++){
		// 		printf("rank %d  k  =  %d got float: %5.2f\t%5.2f\t%5.2f\n", rank, c, data[c*3],data[(c*3)+1],data[(c*3)+2]);
		// }
		// std::cout << std::endl << std::endl;


		while (isSorted != 0) {
			change_even = EvenSort(data, set_pos, global_index_arr, k, rank, size, n);
			MPI_Barrier (MPI_COMM_WORLD);
			change_odd = OddSort(data,set_pos, global_index_arr, k, rank, size, n);
			MPI_Barrier (MPI_COMM_WORLD);
			change = change_even + change_odd;
			MPI_Allreduce(&change, &isSorted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
			// if (isSorted == 0) {
			// 	GarbageRecv(k, rank, size);
			// }
		}
		
		for(int c = 0; c < k; c++){
				printf("rank %d  k  =  %d got float: %5.2f\t%5.2f\t%5.2f\n", rank, c, data[c*3],data[(c*3)+1],data[(c*3)+2]);
		}
		#pragma omp parallel for
		for (unsigned long long i = 0 ; i < k; i++){
			unsigned long long global_idx = global_index_arr[i];
			if (global_idx < n) {
				MPI_File_write_at(out, sizeof(float)*(global_idx), &(data[set_pos[i]]), 1, MPI_FLOAT,MPI_STATUS_IGNORE);
			}
			if (global_idx+1 < n) {
				MPI_File_write_at(out, sizeof(float)*(global_idx+1) , &(data[set_pos[i]+1]), 1, MPI_FLOAT,MPI_STATUS_IGNORE);
			}
		}
		free(data);
		free(set_pos);
		free(global_index_arr);
	}
	MPI_Finalize();
}
