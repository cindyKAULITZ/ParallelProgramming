#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include "debug.h"
#include <emmintrin.h>
#include <smmintrin.h>
//#include <array>
#include <chrono>
#include <numeric>
#include <cstdio>
#include <omp.h>

int totalCompute = 0;
int computeTimes = 0;


__global__ void compute_dist(float * train, float * target, float * dist, int test_idx, int train_end, int cols, int test_end){
    extern __shared__ float sm[];
    float * train_sm = (float *)sm;
    float * test_sm = (float *)&sm[cols * blockDim.y];
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int test_y = test_idx + threadIdx.y;
    if(y < train_end){
            
        for(int i = 0;i < cols;++i){
            train_sm[threadIdx.y * cols + i] = train[y * cols + i];
            test_sm[threadIdx.y * cols + i] = target[test_y * cols + i];
        }
        __syncthreads();
        for(int i = 0; i < blockDim.y; ++i){
            float sum = 0.0;
            if(test_idx + i < test_end){
                for(int j = 0;j < cols;++j){
                    float t0 = train_sm[threadIdx.y * cols + j] - test_sm[i * cols + j];
                    sum += t0 * t0;
                }
                dist[i * train_end + y] = sum;
            }
        }
        
        /*
        for(int i = 0; i < blockDim.y; ++ i){
            float sum = 0.0;
            if(test_idx + i < test_end){
                for(int j = 0;j < cols;++j){
                    float t0 = train[(y) * cols + j] - target[(test_idx + i) * cols + j];
                    sum += t0 * t0;
                }
                dist[i * train_end + y] = sum;
            }
        }
        */

    }
    
}

void print(float * data, int row, int col){
    for(int i = 0;i < row;++i){
        for(int j = 0;j < col;++j){
            std::cout << data[i * col + j] << " ";
        }
        std::cout << "\n";
    }
}

KNNResults KNN::run(int k, DatasetPointer target) {

    
    //std::cout << "is in 1\n";
	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	//squaredDistances: first is the distance; second is the trainExample row
    int tRows = target->rows;
    //std::cout << tRows << std::endl;
    int dRows = data->rows;
    //std::cout << dRows << "is in 2\n";
    int cols = data->cols;
    float * train = data->getMat();
    float * test = target->getMat();
    float * dist = (float *)malloc(sizeof(float) * tRows * dRows);
    long long * idx = (long long *)malloc(sizeof(long long) * tRows * dRows);
    for(long long t = 0;t < tRows;++t){
        std::iota(idx + t * dRows, idx + (t + 1) * dRows, 0);
    }

    int deviceID;
    cudaDeviceProp prop;
	cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID);

    if (!prop.deviceOverlap){
        printf("!prop.deviceOverlap\n");
    }

    float * d_train , * d_test;
    cudaError_t err;
    err = cudaMalloc(&d_train, sizeof(float) * dRows * cols);
    if(err != cudaSuccess){
        std::cout << "fuck you 1\n";
    }
    err = cudaMalloc(&d_test, sizeof(float) * tRows * cols);
    if(err != cudaSuccess){
        std::cout << "fuck you 2\n";
    }
    err = cudaMemcpy(d_train, train, sizeof(float) * dRows * cols, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        std::cout << "fuck you 3\n";
    }
    err = cudaMemcpy(d_test, test, sizeof(float) * tRows * cols, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        std::cout << "fuck you 4\n";
    }

    float * d_dist;
    const int block = 32;
    err = cudaMalloc(&d_dist, sizeof(float) * block * dRows);
    if(err != cudaSuccess){
        std::cout << "fuck you 5\n";
    }

    //std::cout << "is in 3\n";

    int i = 0;
    dim3 numBlock(1, block);
    //std::cout << (tRows + block - 1) / block << "\n";
    dim3 numGrid(1, (dRows + block - 1) / block);
    //std::cout << "is in 4\n";
         
    for(; i < tRows - block; i += block){
        //std::cout << i << " " << tRows << " " << cols << " " << dRows << "\n";
        compute_dist<<<numGrid, numBlock, 2 * block * cols * sizeof(float)>>>(d_train, d_test, d_dist, i, dRows, cols, tRows);
        cudaMemcpy(dist + i * dRows, d_dist, sizeof(float) * block * dRows, cudaMemcpyDeviceToHost);
    }
    

    //cudaMemcpy(dist, d_dist, sizeof(float) * tRows * dRows, cudaMemcpyDeviceToHost);

    //std::cout << "is in\n";
    for(;i < tRows;++i){
#pragma omp parallel for    
        for(int j = 0;j < dRows; ++j){
            float sum = 0.0;
            for(int k = 0;k < cols;++k){
                float t0 = train[j * cols + k] - test[i * cols + k];
                sum += t0 * t0;
            }
            dist[i * dRows + j] = sum;
        }
    }
    
    //print(dist, tRows, dRows);

	//std::pair<double, int> * squaredDistances = new std::pair<double, int>[tRows * dRows];
    std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();



    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
    std::cout << "Compute Distance time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(e - b).count()) / 1000 << "s.\n";

    std::chrono::steady_clock::time_point b1 = std::chrono::steady_clock::now();
//#pragma omp parallel for schedule(static, 256)
    for(int targetExample = 0; targetExample < tRows; targetExample++) {
        std::partial_sort(idx + targetExample * dRows, 
                idx + targetExample * dRows + k,
                idx + (targetExample + 1) * dRows,
                [&dist, &targetExample, &dRows](const int & a,const int & b){
                return dist[targetExample * dRows + a] < dist[targetExample * dRows + b];
                });
    }
    /*
#pragma omp parallel for num_threads(8)
    for(int t = 0; t < tRows * k; t++) {
        int targetExample = t / tRows;
        int i = t % tRows;
        ddist[targetExample * k + i] = dist[idx[i]];
    }
    free(dist);
    dist = ddist;
    */

    std::chrono::steady_clock::time_point e1 = std::chrono::steady_clock::now();
    std::cout << "Compute Sort time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(e1 - b1).count()) / 1000 << "s.\n";


//#pragma omp parallel for schedule(static)
    for(int targetExample = 0; targetExample < tRows; targetExample++) {


        //sort by closest distance

        //count classes of nearest neighbors
        size_t nClasses = target->numLabels;
        int * countClosestClasses = new int[nClasses];
        for(size_t i = 0; i< nClasses; i++)
            countClosestClasses[i] = 0;

        for (int i = 0; i < k; i++)
        {


            int currentClass = data->label(idx[targetExample * dRows + i]);
            countClosestClasses[currentClass]++;
        }

        //result: probability of class K for the example X
        for(size_t i = 0; i < nClasses; i++)
        {
            results->pos(targetExample, i) = ((double)countClosestClasses[i]) / k;
        }
    }

    //copy expected labels:
    for (int i = 0; i < tRows; i++)
        results->label(i) = target->label(i);
    //std::cout << "Average intrinsic time: " << static_cast<double>(totalCompute) / computeTimes  / 1000 << "s.\n";
    free(dist);
    free(idx);
    return KNNResults(results);
}
