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
#include <omp.h>

double intrinSub(std::array<double, 2> a, std::array<double, 2> b){
    __m128d minute = _mm_load_pd(a.data());
    __m128d sub = _mm_load_pd(b.data());
    minute = _mm_sub_pd(minute, sub);
    minute = _mm_mul_pd(minute, minute);
    double ret[2];
    _mm_store_pd(ret, minute);
    return ret[0] + ret[1];
}

double GetSquaredDistance(DatasetPointer train, size_t trainExample, DatasetPointer target, size_t targetExample) {
	assert(train->cols == target->cols);
	double sum = 0;
	double difference;
    size_t col = 0;
	for(; col < train->cols; col+= 2) {
		sum += intrinSub({train->pos(trainExample, col), train->pos(trainExample, col + 1)}, 
                { target->pos(targetExample, col), target->pos(targetExample, col + 1) });
	}
    if(train->cols % 2 == 1){
		difference = train->pos(trainExample, col) - target->pos(targetExample, col);
		sum += difference * difference;
    }
	return sum;
}

KNNResults KNN::run(int k, DatasetPointer target) {



	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	//squaredDistances: first is the distance; second is the trainExample row
    unsigned long long tRows = target->rows;
    unsigned long long dRows = data->rows;
	std::pair<double, int> * squaredDistances = new std::pair<double, int>[tRows * dRows];
    std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
    const unsigned int trainTileSize = 64;
    const unsigned int testTileSize = 128;
    for(unsigned long long testTileBegin = 0; testTileBegin < dRows; testTileBegin += testTileSize){
#pragma omp parallel for num_threads(8)
        for(unsigned long long trainTileBegin = 0; trainTileBegin < dRows; trainTileBegin += trainTileSize){
            for(size_t targetExample = testTileBegin; targetExample < tRows && targetExample < testTileBegin + testTileSize; targetExample++) {
                /*
#ifdef DEBUG_KNN
if (targetExample % 100 == 0)
DEBUGKNN("Target %lu of %lu\n", targetExample, tRows);
#endif
*/
                //Find distance to all examples in the training set
                for (size_t trainExample = trainTileBegin; trainExample < trainTileBegin + trainTileSize && trainExample < dRows; trainExample++) {
                    squaredDistances[targetExample * dRows + trainExample].first = GetSquaredDistance(data, trainExample, target, targetExample);
                    squaredDistances[targetExample * dRows + trainExample].second = trainExample;
                }
            }
        }

    }
    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
    std::cout << "Compute Distance time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(e - b).count()) / 1000 << "s.\n";

    std::chrono::steady_clock::time_point b1 = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(8)
    for(size_t targetExample = 0; targetExample < tRows; targetExample++) {
        sort(squaredDistances + targetExample * dRows, squaredDistances + targetExample * dRows + dRows);
    }
    std::chrono::steady_clock::time_point e1 = std::chrono::steady_clock::now();
    std::cout << "Compute Sort time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(e1 - b1).count()) / 1000 << "s.\n";


#pragma omp parallel for num_threads(8)
	for(size_t targetExample = 0; targetExample < tRows; targetExample++) {


		//sort by closest distance
		
		//count classes of nearest neighbors
		size_t nClasses = target->numLabels;
		int * countClosestClasses = new int[nClasses];
		for(size_t i = 0; i< nClasses; i++)
			 countClosestClasses[i] = 0;

		for (int i = 0; i < k; i++)
		{

			int currentClass = data->label(squaredDistances[targetExample * dRows + i].second);
			countClosestClasses[currentClass]++;
		}

		//result: probability of class K for the example X
		for(size_t i = 0; i < nClasses; i++)
		{
			results->pos(targetExample, i) = ((double)countClosestClasses[i]) / k;
		}
	}

	//copy expected labels:
	for (size_t i = 0; i < target->rows; i++)
		results->label(i) = target->label(i);

	return KNNResults(results);
}
