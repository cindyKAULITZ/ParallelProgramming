#include "knnCuda.cuh"
#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
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


__global__ void CalDistance(int dataCol, int dataRow, int targetRow, int start_round,
							double *train, double *target, 
							double *sq_dist, int *sq_example){

    int x = threadIdx.x;
    int y = threadIdx.y;
    int targetExample = blockIdx.x  * 32 + x;
    int trainExample = (blockIdx.y + start_round) * 32 + y;
    if(trainExample >= dataRow || targetExample >= targetRow){
        return;
    }
    
    // double dist = 0; // = CudaGetSqDist(dataCol, trainExample, targetExample, &train[trainExample*dataCol], &target[targetExample*dataCol]);
    double sum = 0;
	double difference;
    // printf("train->cols = %d\n",cols);
    // #pragma unroll 
	for(int col = 0; col < dataCol; col++) {
        difference = train[trainExample*dataCol+col] - target[targetExample*dataCol+col];
        sum += difference * difference;
        
        // printf("col = %d sum = %f\n",col, sum);
    }
    sq_dist[targetExample * dataRow + trainExample] = sum;
    sq_example[targetExample * dataRow + trainExample] = trainExample;
    // __syncthreads();


}



KNNResults KNNCUDA::run(int k, DatasetPointer target) {
    
	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();
    // printf("first\n");
    
	//squaredDistances: first is the distance; second is the trainExample row
	std::pair<double, int> squaredDistances[data->rows];
	double *squaredDistances_first = (double*)malloc(data->rows* target->rows* sizeof(double));
	int *squaredDistances_second = (int*)malloc(data->rows* target->rows* sizeof(int));
    
	
    
	// DatasetPointer *device_target;
    
    int deviceID;
    cudaDeviceProp prop;
	cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID);
    
    if (!prop.deviceOverlap){
        printf("!prop.deviceOverlap\n");
    }
    // printf("second\n");

    int data_rows = (int)data->rows;
    int data_cols = (int)data->cols;
    int target_rows = (int)target->rows;
    double *data_pos = &data->pos(0,0);
    double *target_pos = &target->pos(0,0);


    // thread limit 1024
    // block limit 65535
    // int block_size = 50000;
    // int thread_size = 1024;

    int B = 32;
    // int B_x = 65535/target_rows - 1; 
    int round_train = (data_rows%B==0)?(data_rows/B):(data_rows/B + 1); 
    int start_round = 0;
    dim3 grid(round_train, target_rows);
    // dim3 grid(target_rows, round_train);
    dim3 blk(B, B);
    // printf("round_train = %d\n", round_train);

    
    // int *e_t;
    // int *e_b;
    // int *r_t;
    // int *r_b;
    
    // if(data_rows < thread_size){
    //     thread_size = data_rows;
    // }
    // if(target_rows < block_size ){
    //     block_size = target_rows;
    // }

    // int ele_per_threads = data_rows/thread_size;
    // int ele_per_blocks = target_rows/block_size;
    // int res_thread = 0;
    // int res_block = 0;

    // if (data_rows%thread_size > 0){
    //     res_thread = data_rows%thread_size;
    // }
    
    // if (target_rows%block_size > 0){
    //     res_block = target_rows%block_size;
    // }
    

    
	int *d_data_rows ;
	int *d_data_cols ;
	int *d_target_rows ;
	double *d_data_pos ;
	double *d_target_pos ;
    
	double *d_sd_dist;
    int *d_sd_example;
    
    
    
    cudaSetDevice(0);
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);

    // cudaMalloc((void **)&e_b,  sizeof(int));
    // cudaMalloc((void **)&e_t,  sizeof(int));
    // cudaMalloc((void **)&r_b,  sizeof(int));
    // cudaMalloc((void **)&r_t,  sizeof(int));

    // cudaMalloc((void **)&d_data_cols,  sizeof(int));
    // cudaMalloc((void **)&d_data_rows,  sizeof(int));
    cudaMalloc(&d_data_pos,  data_cols * data_rows * sizeof(double));
    cudaMalloc(&d_target_pos,  data_cols * target_rows * sizeof(double));
    cudaMalloc(&d_sd_dist,  data_rows * target_rows *  sizeof(double));
    cudaMalloc(&d_sd_example,  data_rows * target_rows * sizeof(int));
    
    // cudaMemcpy(e_b, &ele_per_blocks, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(e_t, &ele_per_threads, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(r_b, &res_block, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(r_t, &res_thread, sizeof(int), cudaMemcpyHostToDevice);

    // cudaMemcpy(d_data_cols, &data_cols, sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_data_rows, &data_rows, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_pos, data_pos,  data_cols * data_rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_pos, target_pos,  data_cols * target_rows * sizeof(double), cudaMemcpyHostToDevice);
    
    // cudaMemcpy(d_sd_dist, 0,  data_rows * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_sd_example, 0,  data_rows * sizeof(int), cudaMemcpyHostToDevice);
    


    // printf("third\n");
    #pragma omp parallel nowait
    for(int i = 0; i < round_train; i++){
        CalDistance<<<grid, blk>>>(data_cols, data_rows, target_rows, (i*round_train), d_data_pos, d_target_pos, d_sd_dist, d_sd_example);
        // start_round += round_train;
    }
    // cudaStreamSynchronize(stream);
    // cudaStreamDestroy(stream);
    // printf("forth\n");
    
	cudaMemcpyAsync(squaredDistances_first , d_sd_dist, data_rows * target_rows * sizeof(double), cudaMemcpyDeviceToHost);
    // printf("fifth-1\n");
	cudaMemcpyAsync(squaredDistances_second , d_sd_example, data_rows * target_rows * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("fifth-2\n");
    // cudaFree(d_data_cols);
    // cudaFree(d_data_rows);
    cudaFree(d_data_pos);
    cudaFree(d_target_pos);
    cudaFree(d_sd_dist);
    cudaFree(d_sd_example);

    #pragma omp parallel for schedule(guided) shared(data, squaredDistances)
	for(size_t targetExample = 0; targetExample < target->rows; targetExample++) {

        
        #pragma unroll
        for (int j = 0; j < data->rows; j++){
            squaredDistances[squaredDistances_second[targetExample *data->rows + j]].first = squaredDistances_first[targetExample *data->rows + j];
            squaredDistances[squaredDistances_second[targetExample *data->rows + j]].second = squaredDistances_second[targetExample *data->rows + j];
            
        }

        // for (int j = 0; j < 5; j++){
        //     if(targetExample < 5){
        //         // printf("squaredDistances_first[targetExample *data->rows + j] = %f\n", squaredDistances_first[targetExample *data->rows + j]);
        //         printf("squaredDistances_second[targetExample *data->rows + j] = %d\n", squaredDistances_second[targetExample *data->rows + j]);
        //         // printf("squaredDistances target(%d) ,train(%d) = %f\n",targetExample,j, squaredDistances[j].first, squaredDistances[j].second);
        //     }
        // }
		//sort by closest distance
        sort(squaredDistances, squaredDistances + data->rows);

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
            // printf("target %d = %d \n", targetExample, results->pos(targetExample, i));
		}
	}

    //copy expected labels:
    #pragma omp parallel for nowait
	for (size_t i = 0; i < target->rows; i++){
		results->label(i) = target->label(i);
    }
   

    free(squaredDistances_first);
    free(squaredDistances_second);

	return KNNResults(results);
}

