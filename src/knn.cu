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
//#include <omp.h>

int totalCompute = 0;
int computeTimes = 0;

__global__ void compute_dist(double * train, double * target, double * dist, int idx, int end, int rows, int trRows){
    extern __shared__ double sm[];
    double * train_sm = sm;
    double * test_sm = sm + rows * blockDim.y;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int train_y = idx + threadIdx.y;
    if(train_y == 0 && y == 0){
        printf("is in\n");
    }
    if(y < end){
        for(int i = 0;i < rows;++i){
            train_sm[threadIdx.y * rows + i] = train[train_y * rows + i];
            test_sm[threadIdx.y * rows + i] = target[y * rows + i];
        }
        __syncthreads();
        for(int i = 0; i < blockDim.y; ++ i){
            double sum = 0.0;
            for(int j = 0;j < rows;++j){
                double t0 = train_sm[i * rows + j] - test_sm[i * rows + j];
                sum += t0 * t0;
            }
            dist[y * trRows + train_y] = 11.3;
        }
    }
    dist[threadIdx.y] = 11.3;
}

void print(double * data, int row, int col){
    for(int i = 0;i < row;++i){
        for(int j = 0;j < col;++j){
            std::cout << data[i * col + j] << " ";
        }
        std::cout << "\n";
    }
}

KNNResults KNN::run(int k, DatasetPointer target) {

    
    std::cout << "is in 1\n";
	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	//squaredDistances: first is the distance; second is the trainExample row
    int tRows = target->rows;
    int dRows = data->rows;
    int cols = data->cols;
    double * train = data->getMat();
    double * test = target->getMat();
    double * dist = (double *)malloc(sizeof(double) * tRows * dRows);
    long long * idx = (long long *)malloc(sizeof(long long) * tRows * dRows);
    for(long long t = 0;t < tRows;++t){
        std::iota(idx + t * dRows, idx + (t + 1) * dRows, 0);
    }

    std::cout << "is in 2\n";
    int deviceID;
    cudaDeviceProp prop;
	cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID);

    if (!prop.deviceOverlap){
        printf("!prop.deviceOverlap\n");
    }

    double * d_train , * d_test;
    cudaMalloc(&d_train, sizeof(double) * dRows * cols);
    cudaMalloc(&d_test, sizeof(double) * tRows * cols);
    cudaMemcpy(d_train, train, sizeof(double) * dRows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test, test, sizeof(double) * tRows * cols, cudaMemcpyHostToDevice);

    double * d_dist;
    const int block = 1024;
    cudaMalloc(&d_dist, sizeof(double) * tRows * dRows);

    std::cout << "is in 3\n";

    int i = 0;
    dim3 numBlock(1, block);
    dim3 numGrid(1, (tRows + block - 1) / block);
    std::cout << "is in 4\n";
       
    for(; i < dRows - block; i += block){
        compute_dist<<<numGrid, numBlock, 2 * block * cols>>>(d_train, d_test, d_dist, i, tRows, cols, dRows);
    }
    
    



    cudaMemcpy(dist, d_dist, sizeof(double) * tRows * dRows, cudaMemcpyDeviceToHost);

    std::cout << "is in\n";

    for(;i < dRows;++i){
        for(int j = 0;j < tRows; ++j){
            double sum = 0.0;
            for(int k = 0;k < cols;++k){
                double t0 = train[i * cols + k] - test[j * cols + k];
                sum += t0 * t0;
            }
            dist[j * dRows + i] = sum;
        }
    }
    print(dist, tRows, dRows);

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
    std::cout << "Average intrinsic time: " << static_cast<double>(totalCompute) / computeTimes  / 1000 << "s.\n";
    free(dist);
    free(idx);
    return KNNResults(results);
}
