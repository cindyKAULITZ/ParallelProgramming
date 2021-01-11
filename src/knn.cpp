#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include "debug.h"
#include <cstdio>
#include <cstdlib>
#include <chrono> 

using namespace std;

double GetSquaredDistance(DatasetPointer train, size_t trainExample, DatasetPointer target, size_t targetExample) {
	assert(train->cols == target->cols);
	double sum = 0;
	double difference;
	// printf("train->cols = %d\n",train->cols);
	for(size_t col = 0; col < train->cols; col++) {
		difference = train->pos(trainExample, col) - target->pos(targetExample, col);
		sum += difference * difference;
	}
	return sum;
}

KNNResults KNN::run(int k, DatasetPointer target) {

	double copyTime = 0.0;
    double sortTime = 0.0;
    double processTime = 0.0;

    std::chrono::steady_clock::time_point p_0 = std::chrono::steady_clock::now();
    
	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	//squaredDistances: first is the distance; second is the trainExample row
	std::pair<double, int> squaredDistances[data->rows];
	double total = 0.0;
	std::chrono::steady_clock::time_point p_1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> p_t = p_1 - p_0;
    processTime+=p_t.count();

	for(size_t targetExample = 0; targetExample < target->rows; targetExample++) {

// #ifdef DEBUG_KNN
// 		if (targetExample % 100 == 0)
// 				DEBUGKNN("Target %lu of %lu\n", targetExample, target->rows);
// #endif
		//Find distance to all examples in the training set
		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		for (size_t trainExample = 0; trainExample < data->rows; trainExample++) {
				squaredDistances[trainExample].first = GetSquaredDistance(data, trainExample, target, targetExample);
				squaredDistances[trainExample].second = trainExample;
			}
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		std::chrono::duration<double> dt = t1 - t0;
		total += (double)dt.count();

		std::chrono::steady_clock::time_point t_sort1 = std::chrono::steady_clock::now();

		sort(squaredDistances, squaredDistances + data->rows);
		std::chrono::steady_clock::time_point t_sort0 = std::chrono::steady_clock::now();
        std::chrono::duration<double> t_sort = t_sort0 - t_sort1;
        sortTime += t_sort.count();

		//count classes of nearest neighbors
		p_0 = std::chrono::steady_clock::now();
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
		p_1 = std::chrono::steady_clock::now();
        p_t = p_1 - p_0;
        processTime += p_t.count();
	}
	
    p_0 = std::chrono::steady_clock::now();
	//copy expected labels:
	for (size_t i = 0; i < target->rows; i++){
		results->label(i) = target->label(i);
	}
	p_1 = std::chrono::steady_clock::now();
	p_t = p_1 - p_0;
	processTime += p_t.count();

	printf("CalDistance Time %lfs\n", total);
    printf("Sort took %lfs\n", sortTime); 
    printf("Other Process took %lfs\n", processTime); 

	return KNNResults(results);
}
