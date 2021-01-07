#include "knn.h"
#include <cassert>
#include <algorithm>
#include "dataset.h"
#include <iostream>
#include "debug.h"
#include <chrono>

static int total_dist_compute_time = 0;
static int total_sort_time = 0;
static int total_find_k_time = 0;

double GetSquaredDistance(DatasetPointer train, size_t trainExample, DatasetPointer target, size_t targetExample) {
	assert(train->cols == target->cols);
	double sum = 0;
	double difference;
	for(size_t col = 0; col < train->cols; col++) {
		difference = train->pos(trainExample, col) - target->pos(targetExample, col);
		sum += difference * difference;
	}
	return sum;
}

KNNResults KNN::run(int k, DatasetPointer target) {

    std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();

	DatasetPointer results(new dataset_base(target->rows,target->numLabels, target->numLabels));
	results->clear();
    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    std::cout << "Total fill to zero time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - b).count()) / 1000 << "s.\n";

	//squaredDistances: first is the distance; second is the trainExample row
	std::pair<double, int> squaredDistances[data->rows];

	for(size_t targetExample = 0; targetExample < target->rows; targetExample++) {
/*
#ifdef DEBUG_KNN
		if (targetExample % 100 == 0)
				DEBUGKNN("Target %lu of %lu\n", targetExample, target->rows);
#endif
*/
		//Find distance to all examples in the training set
        std::chrono::steady_clock::time_point c_start = std::chrono::steady_clock::now();
		for (size_t trainExample = 0; trainExample < data->rows; trainExample++) {
				squaredDistances[trainExample].first = GetSquaredDistance(data, trainExample, target, targetExample);
				squaredDistances[trainExample].second = trainExample;
		}
        std::chrono::steady_clock::time_point c_end = std::chrono::steady_clock::now();
        total_dist_compute_time += std::chrono::duration_cast<std::chrono::milliseconds>(c_end - c_start).count();

		//sort by closest distance
        std::chrono::steady_clock::time_point s_start = std::chrono::steady_clock::now();
		sort(squaredDistances, squaredDistances + data->rows);
        std::chrono::steady_clock::time_point s_end = std::chrono::steady_clock::now();
        total_sort_time += std::chrono::duration_cast<std::chrono::milliseconds>(s_end - s_start).count();
		
		//count classes of nearest neighbors
        std::chrono::steady_clock::time_point f_start = std::chrono::steady_clock::now();
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
        std::chrono::steady_clock::time_point f_end = std::chrono::steady_clock::now();
        total_find_k_time += std::chrono::duration_cast<std::chrono::milliseconds>(f_end - f_start).count();
	}
    std::cout << "Total compute time: " << static_cast<double>(total_dist_compute_time) / 1000 << "s.\n";
    std::cout << "Total sort time: " << static_cast<double>(total_sort_time) / 1000 << "s.\n";
    std::cout << "Total find k & class time: " << static_cast<double>(total_find_k_time) / 1000 << "s.\n";

	//copy expected labels:
	for (size_t i = 0; i < target->rows; i++)
		results->label(i) = target->label(i);

    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
    std::cout << "Total knn time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(e - b).count()) / 1000 << "s.\n";
	return KNNResults(results);
}
