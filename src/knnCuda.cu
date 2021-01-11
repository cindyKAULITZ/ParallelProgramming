#include "knnCuda.cuh"
#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include <algorithm>
#include "debug.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <chrono> 
#include <omp.h>

using namespace std;

// double GetSquaredDistance(DatasetPointer train, size_t trainExample, DatasetPointer target, size_t targetExample) {
// 	assert(train->cols == target->cols);
// 	double sum = 0;
// 	double difference;
// 	// printf("train->cols = %d\n",train->cols);
// 	for(size_t col = 0; col < train->cols; col++) {
// 		difference = train->pos(trainExample, col) - target->pos(targetExample, col);
// 		sum += difference * difference;
// 	}
// 	return sum;
// }
__device__ double CudaGetSqDist(int cols, int trainExample, int targetExample, double *train, double *target) {
	double sum = 0;
    double difference;
    
	for(int col = 0; col < cols; col++) {
        difference = train[col] - target[col];
        sum += difference * difference;
        
    }
	return sum;
}

__global__ void CalDistance(int dataCol, int dataRow, int e_t, int e_b, int r_t, int r_b,
							double *train, double *target, 
							double *sq_dist, int *sq_example){
	
    
	// int targetExample = blockIdx.x;
    // int trainExample = threadIdx.x;
    int cols = dataCol;
    int start_b = blockIdx.x * e_b;
    int num_b = e_b;
    if(r_b > 0 && blockIdx.x < r_b-1){
        start_b = blockIdx.x * (e_b+1);
        num_b = e_b+1;
    }
    int start_t = threadIdx.x * e_t;
    int num_t = e_t;
    if(r_t > 0 && threadIdx.x < r_t-1){
        start_t = threadIdx.x * (e_t+1);
        num_t = e_t+1;
    }
    // printf("start_b = %d, start_t= %d\n", start_b, start_t);
    // printf("blockIdx.x = %d, threadIdx.x = %d\n", blockIdx.x, threadIdx.x);

    for(int targetExample = start_b; targetExample < start_b+num_b; targetExample++ ){
        for(int trainExample = start_t; trainExample < start_t+num_t; trainExample++){
            double dist = CudaGetSqDist(dataCol, trainExample, targetExample, &train[trainExample*cols], &target[targetExample*cols]);
            sq_dist[targetExample * dataRow + trainExample] = dist;
            sq_example[targetExample * dataRow + trainExample] = trainExample;
        }
    }

    // double dist = CudaGetSqDist(dataCol, trainExample, targetExample, &train[trainExample*cols], &target[targetExample*cols]);
    // sq_dist[targetExample * dataRow + trainExample] = dist;
    // sq_example[targetExample * dataRow + trainExample] = trainExample;
    // __syncthreads();


}



KNNResults KNNCUDA::run(int k, DatasetPointer target, int b_s) {
    double copyTime = 0.0;
    double sortTime = 0.0;
    double processTime = 0.0;

    std::chrono::steady_clock::time_point p_0 = std::chrono::steady_clock::now();
    
	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();
    
	//squaredDistances: first is the distance; second is the trainExample row
	std::pair<double, int> squaredDistances[data->rows];
	double *squaredDistances_first = (double*)malloc(data->rows* target->rows* sizeof(double));
	int *squaredDistances_second = (int*)malloc(data->rows* target->rows* sizeof(int));
    
	
    
    int deviceID;
    cudaDeviceProp prop;
	cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID);
    
    if (!prop.deviceOverlap){
        printf("!prop.deviceOverlap\n");
    }

    int data_rows = (int)data->rows;
    int data_cols = (int)data->cols;
    int target_rows = (int)target->rows;
    double *data_pos = &data->pos(0,0);
    double *target_pos = &target->pos(0,0);


    // thread limit 1024
    // block limit 65535
    int block_size = b_s;
    int thread_size = 1024;

    
    int *e_t;
    int *e_b;
    int *r_t;
    int *r_b;
    
    if(data_rows < thread_size){
        thread_size = data_rows;
    }
    if(target_rows < block_size ){
        block_size = target_rows;
    }

    int ele_per_threads = data_rows/thread_size;
    int ele_per_blocks = target_rows/block_size;
    int res_thread = 0;
    int res_block = 0;

    if (data_rows%thread_size > 0){
        res_thread = data_rows%thread_size;
    }
    
    if (target_rows%block_size > 0){
        res_block = target_rows%block_size;
    }
    
    // printf("block_size %d, thread_size %d\n", block_size, thread_size);
    // printf("e_t %d, e_b %d, r_t %d, r_b %d\n", ele_per_threads,ele_per_blocks, res_thread, res_block);
    
    
	// int *d_data_rows ;
	// int *d_data_cols ;
	// int *d_target_rows ;
	double *d_data_pos ;
	double *d_target_pos ;
    
	double *d_sd_dist;
    int *d_sd_example;
    std::chrono::steady_clock::time_point p_1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> p_t = p_1 - p_0;
    processTime+=p_t.count();
    
    
    
    cudaSetDevice(0);

    cudaMalloc(&d_data_pos,  data_cols * data_rows * sizeof(double));
    cudaMalloc(&d_target_pos,  data_cols * target_rows * sizeof(double));
    cudaMalloc(&d_sd_dist,  data_rows * target_rows *  sizeof(double));
    cudaMalloc(&d_sd_example,  data_rows * target_rows * sizeof(int));

    cudaMemcpy(d_data_pos, data_pos,  data_cols * data_rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_pos, target_pos,  data_cols * target_rows * sizeof(double), cudaMemcpyHostToDevice);
    

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
	CalDistance<<<block_size, thread_size>>>(data_cols, data_rows, ele_per_threads, ele_per_blocks, res_thread, res_block, d_data_pos, d_target_pos, d_sd_dist, d_sd_example);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    printf("CalDistance Time %gs (%lfs)\n", dt.count(), dt.count()); 

	cudaMemcpy(squaredDistances_first , d_sd_dist, data_rows * target_rows * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(squaredDistances_second , d_sd_example, data_rows * target_rows * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data_pos);
    cudaFree(d_target_pos);
    cudaFree(d_sd_dist);
    cudaFree(d_sd_example);
    

    #pragma omp parallel for shared(data, squaredDistances)
	for(size_t targetExample = 0; targetExample < target->rows; targetExample++) {

        std::chrono::steady_clock::time_point t_cpy1 = std::chrono::steady_clock::now();
        #pragma unroll
        for (int j = 0; j < data->rows; j++){
            int idx = squaredDistances_second[targetExample *data->rows + j];
            squaredDistances[squaredDistances_second[idx]].first = squaredDistances_first[targetExample *data->rows + j];
            squaredDistances[squaredDistances_second[idx]].second = idx;
            
        }
        std::chrono::steady_clock::time_point t_cpy0 = std::chrono::steady_clock::now();
        std::chrono::duration<double> t_cpy = t_cpy0 - t_cpy1;
        copyTime += t_cpy.count(); 

		//sort by closest distance
        std::chrono::steady_clock::time_point t_sort1 = std::chrono::steady_clock::now();

        // sort(squaredDistances, squaredDistances + data->rows);
        partial_sort(squaredDistances, squaredDistances + k, squaredDistances + data->rows);
        std::chrono::steady_clock::time_point t_sort0 = std::chrono::steady_clock::now();
        std::chrono::duration<double> t_sort = t_sort0 - t_sort1;
        sortTime += t_sort.count();
        

        p_0 = std::chrono::steady_clock::now();
		//count classes of nearest neighbors
		size_t nClasses = target->numLabels;
		int countClosestClasses[nClasses];
		for(size_t i = 0; i< nClasses; i++)
			 countClosestClasses[i] = 0;

		for (int i = 0; i < k; i++)
		{

			int currentClass = data->label(squaredDistances[i].second);
			countClosestClasses[currentClass]++;
		}

		//result: probability of class K for the example X
		for(size_t i = 0; i < nClasses; i++)
		{
            results->pos(targetExample, i) = ((double)countClosestClasses[i]) / k;
        }
        p_1 = std::chrono::steady_clock::now();
        p_t = p_1 - p_0;
        processTime += p_t.count();
        
    }

    //copy expected labels:
    p_0 = std::chrono::steady_clock::now();
    #pragma omp parallel for nowait
	for (size_t i = 0; i < target->rows; i++){
		results->label(i) = target->label(i);
    }
    p_1 = std::chrono::steady_clock::now();
    p_t = p_1 - p_0;
    processTime += p_t.count();
   
    
    printf("Sort took %lfs\n", sortTime); 
    printf("Other Process took %lfs\n", processTime); 

    printf("Copy took %lfs\n", copyTime);

    free(squaredDistances_first);
    free(squaredDistances_second);

	return KNNResults(results);
}

