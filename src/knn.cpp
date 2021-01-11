#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include "debug.h"
#include <emmintrin.h>
#include <smmintrin.h>
#include <array>
#include <chrono>
#include <numeric>
#include <thread>
#include <mutex>
#include <omp.h>

int totalCompute = 0;
int computeTimes = 0;
int comunicateTimes = 0;
int i = 0;
int j;

static double * train;
static double * test;
static double * dist;

static int tRows;
static int dRows;
static int cols;

int chunck = 1;

std::mutex mu1;

void computeDist(int beg){
    /*
    int local_i = beg;
    for(; local_i < local_i + chunck && local_i < tRows; ++ local_i){
        for(int lj = 0; lj < dRows; ++lj){
            double sum = 0.0;
            for(int lc = 0; lc < cols;++lc){
                double t0 = train[lj * cols + lc] - test[local_i * cols + lc];
                sum += t0 * t0;
            }
            dist[local_i * dRows + lj] = sum;
        }
    }
    */
    
    int local_i = beg;
    while(local_i < tRows){
        //printf("%d\n", local_i);
        
        for(; local_i < local_i + chunck && local_i < tRows; ++ local_i){
            for(int lj = 0; lj < dRows; ++lj){
                double sum = 0.0;
                for(int lc = 0; lc < cols;++lc){
                    double t0 = train[lj * cols + lc] - test[local_i * cols + lc];
                    sum += t0 * t0;
                }
                dist[local_i * dRows + lj] = sum;
            }
        }
        
        std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
        mu1.lock(); 
        if(local_i == i){
            i += chunck;
        }
        local_i = i;
        i += chunck;
        std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
        computeTimes += std::chrono::duration_cast<std::chrono::milliseconds>(e - b).count();
        mu1.unlock();
    }
}

KNNResults KNN::run(int k, DatasetPointer target) {

    std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	//squaredDistances: first is the distance; second is the trainExample row
    tRows = target->rows;
    dRows = data->rows;
    cols = data->cols;
    dist = (double *)malloc(sizeof(double) * tRows * dRows);
    train = data->getMat();
    test = target->getMat();
    long long * idx = (long long *)malloc(sizeof(long long) * tRows * dRows);

    const int num_cores = std::thread::hardware_concurrency();

    std::cout << num_cores << "\n";

    for(long long t = 0;t < tRows;++t){
        std::iota(idx + t * dRows, idx + (t + 1) * dRows, 0);
    }
	//std::pair<double, int> * squaredDistances = new std::pair<double, int>[tRows * dRows];
    //for(unsigned long long testTileBegin = 0; testTileBegin < dRows; testTileBegin += testTileSize){
//#pragma omp parallel for schedule(static) num_threads(8)
        //for(unsigned long long trainTileBegin = 0; trainTileBegin < dRows; trainTileBegin += trainTileSize){
    std::thread pool[12];
    /*
    int dataBegin[12];
    int sizeAry[12];
    int size = tRows / num_cores;
    int re = tRows % num_cores;
    for(int i = 0;i < num_cores;++i){
        dataBegin[i] = 0;
        sizeAry[i] = size;
    }

    for(int i = 1;i < num_cores;++i){
        if(re > 0){
            dataBegin[i] = dataBegin[i - 1] + size + 1;
            sizeAry[i] += 1;
            re -= 1;
        }
        else{
            dataBegin[i] = dataBegin[i - 1] + size;
        }
    }
    
    */
    i = num_cores < tRows ? num_cores : 0;
    for(int i = 0;i < num_cores;++i){
        pool[i] = std::thread(computeDist, i);
    }
    for(int i = 0;i < num_cores;++i){
        pool[i].join();
    }
    /*
    for (int trainExample = 0; trainExample < dRows; trainExample++) {
        
        for(int targetExample = 0; targetExample < tRows; targetExample++) {
            int d = 0;
            double sum = 0;
            for(; d < cols; ++d){
                double t0 = data->pos(trainExample, d) - target->pos(targetExample, d);
                sum += t0 * t0;
            }
            dist[targetExample * dRows + trainExample] = sum;
        }
        
    }
    */
    //}

    //}
    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
    std::cout << "Compute Distance time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(e - b).count()) / 1000 << "s.\n";

    double * ddist = dist;
    int ddRows = dRows;
    std::chrono::steady_clock::time_point b1 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static, 256)
    for(int targetExample = 0; targetExample < tRows; targetExample++) {
        std::partial_sort(idx + targetExample * dRows, 
                idx + targetExample * dRows + k,
                idx + (targetExample + 1) * dRows,
                [&ddist, &targetExample, &ddRows](const int & a,const int & b){
                return ddist[targetExample * ddRows + a] < ddist[targetExample * ddRows + b];
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


#pragma omp parallel for schedule(static)
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
    std::cout << "Comunication time: " << static_cast<double>(computeTimes) / 1000 << "s.\n";
    free(dist);
    free(idx);
    return KNNResults(results);
}
