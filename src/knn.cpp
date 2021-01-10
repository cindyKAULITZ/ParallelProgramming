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


	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();

	//squaredDistances: first is the distance; second is the trainExample row
	std::pair<double, int> squaredDistances[data->rows];
	
	for(size_t targetExample = 0; targetExample < target->rows; targetExample++) {

// #ifdef DEBUG_KNN
// 		if (targetExample % 100 == 0)
// 				DEBUGKNN("Target %lu of %lu\n", targetExample, target->rows);
// #endif
		//Find distance to all examples in the training set
		for (size_t trainExample = 0; trainExample < data->rows; trainExample++) {
				squaredDistances[trainExample].first = GetSquaredDistance(data, trainExample, target, targetExample);
				squaredDistances[trainExample].second = trainExample;
				// if( targetExample == 3  ){
                // // printf("start_b %d, num_b %d, start_t %d, num_t %d\n",start_b, num_b, start_t, num_t);
                // 	printf("CalDistance target(%d) ,train(%d) = %f \n",targetExample,trainExample, squaredDistances[trainExample].first);

            	// }
				// if( squaredDistances[trainExample].first == 0)
				// 	printf("CalDistance target(%d) ,train(%d) = %f \n",targetExample,trainExample, squaredDistances[trainExample].first);
		}
		// for (int j = 0; j < 1; j++){
        //     printf("squaredDistances target(%d) ,train(%d) = %f (%d) \n",targetExample,j, squaredDistances[j].first, squaredDistances[j].second);
        // }
		// for (int j = 0; j < 1; j++){
        //     printf("squaredDistances target(%d) ,train(%d) = %f\n",targetExample,j, squaredDistances[j].first, squaredDistances[j].second);
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
	for (size_t i = 0; i < target->rows; i++){
		results->label(i) = target->label(i);
	}

	return KNNResults(results);
}
