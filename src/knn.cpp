#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include "debug.h"
#include <emmintrin.h>
#include <smmintrin.h>
#include <array>
#include <Halide.h>
#include <chrono>
#include <numeric>
#include <omp.h>

int totalCompute = 0;
int computeTimes = 0;

void transformMatrix(double * dst,DatasetPointer src, int rows, int cols){
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            dst[j * rows + i] = src->pos(i, j);
        }
    }
}

KNNResults KNN::run(int k, DatasetPointer target) {

	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	//squaredDistances: first is the distance; second is the trainExample row
    int tRows = target->rows;
    int dRows = data->rows;
    int cols = data->cols;
    double * dist = (double *)malloc(sizeof(double) * tRows * dRows);
    double * trainMat = (double *)malloc(sizeof(double) * dRows * cols);
    transformMatrix(trainMat, data, dRows, cols);
    double * testMat = target->getMat();
    long long * idx = (long long *)malloc(sizeof(long long) * tRows * dRows);
    for(long long t = 0;t < tRows;++t){
        std::iota(idx + t * dRows, idx + (t + 1) * dRows, 0);
    }
	//std::pair<double, int> * squaredDistances = new std::pair<double, int>[tRows * dRows];
    std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
    //for(unsigned long long testTileBegin = 0; testTileBegin < dRows; testTileBegin += testTileSize){
//#pragma omp parallel for schedule(static) num_threads(8)
        //for(unsigned long long trainTileBegin = 0; trainTileBegin < dRows; trainTileBegin += trainTileSize){
    Halide::Buffer<double> m1(testMat, cols, tRows, "train");
    Halide::Buffer<double> m2(trainMat, dRows, cols, "test");
    Halide::Buffer<double> dist_B(dist, dRows, tRows, "dist");
    Halide::Var x("train-row"), y("test-row"), kk("col");
    Halide::Func mul("multiplication"), mat1("test-mat"), mat2("train-mat");
    //Halide::Func cube("cube");
    Halide::RDom r(0, cols, "num-cols");
    //cube(x, y, kk) = m1(kk, y) - m2(x, kk);
    //cube.print_loop_nest();
    //printf("\n");
    //mat1(kk, y) = m1(kk, y);
    //mat2(x, kk) = m2(x, kk);
    mul(x, y) = Halide::cast<double>(0);
    mul(x, y) += (m1(r, y) - m2(x, r)) * (m1(r, y) - m2(x, r));
    //mul(x, y) += cube(x, y, r) * cube(x, y, r);
    //Halide::Var yin, yout;
    //mul.split(y, yout, yin, 256);
    //mul.reorder(yout, x, yin);
    //mul.parallel(yout);
    //cube.compute_at(mul, y);
    //cube.parallel(x).vectorize(kk, 4);
    
    //Halide::Var yinner, youter;
    //mul.split(x, xouter, xinner, 4);
    //mul.vectorize(xinner);
    //
    /*
    cube.compute_at(mul, x);
    mul.split(y, youter, yinner, 256);
    mul.reorder(youter, x, yinner);
    mul.parallel(youter);
    mul.vectorize(yinner);
    */

    mul.print_loop_nest();
    printf("\n");

    //Halide::Var yinner, youter;
    //mul.split(y, youter, yinner, 4);
    //mul.unroll(yinner);
    //mul.parallel(x);


    mul.realize(dist_B);
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
