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
#include <omp.h>

int totalCompute = 0;
int computeTimes = 0;

KNNResults KNN::run(int k, DatasetPointer target) {

	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	//squaredDistances: first is the distance; second is the trainExample row
    int tRows = target->rows;
    int dRows = data->rows;
    int cols = data->cols;
    double * dist = (double *)malloc(sizeof(double) * tRows * dRows);
    long long * idx = (long long *)malloc(sizeof(long long) * tRows * dRows);
    for(long long t = 0;t < tRows;++t){
        std::iota(idx + t * dRows, idx + (t + 1) * dRows, 0);
    }
	//std::pair<double, int> * squaredDistances = new std::pair<double, int>[tRows * dRows];
    std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
    //for(unsigned long long testTileBegin = 0; testTileBegin < dRows; testTileBegin += testTileSize){
//#pragma omp parallel for schedule(static) num_threads(8)
        //for(unsigned long long trainTileBegin = 0; trainTileBegin < dRows; trainTileBegin += trainTileSize){
    for(int targetExample = 0; targetExample < tRows; targetExample++) {
#pragma omp parallel for schedule(static, 256)
        for (int trainExample = 0; trainExample < dRows; trainExample++) {
            /*
#ifdef DEBUG_KNN
if (targetExample % 100 == 0)
DEBUGKNN("Target %lu of %lu\n", targetExample, tRows);
#endif
*/
            //Find distance to all examples in the training set
            //dist[targetExample * dRows + trainExample] = GetSquaredDistance(data, trainExample, target, targetExample);
            //squaredDistances[targetExample * dRows + trainExample].second = trainExample;
            int d = 0;
            double sum = 0;
            for(; d < cols; ++d){
                double t0 = data->pos(trainExample, d) - target->pos(targetExample, d);
                sum += t0 * t0;
            }
            dist[targetExample * dRows + trainExample] = sum;
        }
    }
    //}

    //}
    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
    std::cout << "Compute Distance time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(e - b).count()) / 1000 << "s.\n";

    std::chrono::steady_clock::time_point b1 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static, 256)
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
    std::cout << "Average intrinsic time: " << static_cast<double>(totalCompute) / computeTimes  / 1000 << "s.\n";
    free(dist);
    free(idx);
    return KNNResults(results);
}
