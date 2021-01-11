#include <iostream>
#include "knn.h"
#include "knnCuda.cuh"
#include "ReadDataset.h"
#include "dataset.h"
#include "Preprocessing.h"
#include <new>
#include <cstring>
#include <chrono> 
// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <cuda_runtime.h>
// #include <cuda.h>
#include <cstdio>
#include <cstdlib>

using namespace std;

const int nLabels = 11;
// const int nLabels = 10;

void runKnn(char *trainFile, char *testFile, int k, int cuda, int b_s) {
	cout << "Reading train" <<endl;
	DatasetPointer train = ReadDataset::read(trainFile, nLabels);
	cout << "Reading test" <<endl;
	DatasetPointer test = ReadDataset::read(testFile, nLabels);
	// cout << "MeanNormalize" <<endl;
	MatrixPointer meanData = MeanNormalize(train);

	if(cuda == 1){
		cout << "KNNCUDA knn(train)" <<endl;
		KNNCUDA knn(train);
		// cout << "ApplyMeanNormalization" <<endl;
		ApplyMeanNormalization(test, meanData);

		// cout << "knn.run(k, test);" <<endl;
		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(); 
		KNNResults rawResults = knn.run(k, test, b_s);
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		std::chrono::duration<double> dt = t1 - t0;
    	printf("Run cuda KNN Total time %gs\n", dt.count()); 

		cout << "Consolidating results" << endl;
		SingleExecutionResults top1 = rawResults.top1Result();
		SingleExecutionResults top2 = rawResults.topXResult(2);
		SingleExecutionResults top3 = rawResults.topXResult(3);

		printf("Success Rate: %lf, Rejection Rate: %lf\n", top1.successRate(), top1.rejectionRate());
		printf("Top 2 Success Rate: %lf\n", top2.successRate());
		printf("Top 3 Success Rate: %lf\n", top3.successRate());
		// printf("Confusion matrix:\n");

		// MatrixPointer confusionMatrix = rawResults.getConfusionMatrix();

		// for(size_t i = 0; i< confusionMatrix->rows; i++) {
		// 	for(size_t j = 0; j< confusionMatrix->cols; j++) {
		// 		if (j!=0) printf(",");
		// 		printf("%d", (int)confusionMatrix->pos(i,j));
		// 	}
		// 	printf("\n");
		// }
	}else{
		cout << "KNN knn(train)" <<endl;
		KNN knn(train);
		// cout << "ApplyMeanNormalization" <<endl;
		ApplyMeanNormalization(test, meanData);

		// cout << "knn.run(k, test);" <<endl;
		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(); 
		KNNResults rawResults = knn.run(k, test);
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		std::chrono::duration<double> dt = t1 - t0;
    	printf("Run KNN Total time %gs\n", dt.count()); 
		cout << "Consolidating results" << endl;
		SingleExecutionResults top1 = rawResults.top1Result();
		SingleExecutionResults top2 = rawResults.topXResult(2);
		SingleExecutionResults top3 = rawResults.topXResult(3);

		printf("Success Rate: %lf, Rejection Rate: %lf\n", top1.successRate(), top1.rejectionRate());
		printf("Top 2 Success Rate: %lf\n", top2.successRate());
		printf("Top 3 Success Rate: %lf\n", top3.successRate());
		// printf("Confusion matrix:\n");

		// MatrixPointer confusionMatrix = rawResults.getConfusionMatrix();

		// for(size_t i = 0; i< confusionMatrix->rows; i++) {
		// 	for(size_t j = 0; j< confusionMatrix->cols; j++) {
		// 		if (j!=0) printf(",");
		// 		printf("%d", (int)confusionMatrix->pos(i,j));
		// 	}
		// 	printf("\n");
		// }
	}

}


void findBestK(char *trainFile, int cuda, int b_s) {
	cout << "Reading train" <<endl;
	DatasetPointer data = ReadDataset::read(trainFile, nLabels);

	DatasetPointer train, valid;

	data->splitDataset(train, valid, 0.8);

	MatrixPointer meanData = MeanNormalize(train);
	ApplyMeanNormalization(valid, meanData);

	if(cuda == 1){
		
		KNNCUDA knn(train);
		
		double bestSuccessRate = 0;
		int bestK = 0;
		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		for(int k=1; k<=20; k++) {
			printf("Trying cuda K = %d ... ",k);
			KNNResults res = knn.run(k, valid, b_s);
			// printf("res ... ");
			double currentSuccess = res.top1Result().successRate();
			// printf("currentSuccess ... ");
			if (currentSuccess > bestSuccessRate) {
				bestSuccessRate = currentSuccess;
				bestK = k;
			}
			printf("%lf\n", currentSuccess);
		}
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		std::chrono::duration<double> dt = t1 - t0;
    	printf("Run cuda KNN Total time %gs\n", dt.count()); 

		printf("Best K: %d. Success rate in validation set: %lf\n", bestK, bestSuccessRate);
		
	
	}else{
		KNN knn(train);

		double bestSuccessRate = 0;
		int bestK = 0;
		std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
		for(int k=1; k<=20; k++) {
			printf("Trying K = %d ... ",k);
			KNNResults res = knn.run(k, valid);
			// printf("res ... ");
			double currentSuccess = res.top1Result().successRate();
			// printf("currentSuccess ... ");
			if (currentSuccess > bestSuccessRate) {
				bestSuccessRate = currentSuccess;
				bestK = k;
			}
			printf("%lf\n", currentSuccess);
		}
		std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
		std::chrono::duration<double> dt = t1 - t0;
    	printf("Run KNN Total time %gs\n", dt.count()); 

		printf("Best K: %d. Success rate in validation set: %lf\n", bestK, bestSuccessRate);
	}
	
}

void printUsageAndExit(char **argv);


int main(int argc, char **argv)
{
	if (argc != 5 && argc != 7) {
		printUsageAndExit(argv);
	}

	if (strcmp(argv[1], "run") == 0) {
		runKnn(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
	}
	else if (strcmp(argv[1], "findbest") == 0) {
		findBestK(argv[2], atoi(argv[3]), atoi(argv[4]));
	}
	else
		printUsageAndExit(argv);
}

void printUsageAndExit(char **argv) {
	printf("Usage:\n"
		"%s run <train dataset> <test dataset> <k> <cuda:1 seq:0> : run KNN\n"
		"%s findbest <train dataset> <cuda:1 seq:0>: Find K that minimizes error (1~20)\n",argv[0], argv[0]);
	exit(1);
}
