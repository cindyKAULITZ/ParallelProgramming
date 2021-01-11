#ifndef KNNCUDA_H
#define KNNCUDA_H

#include "dataset.h"
#include "knn.h"

// class SingleExecutionResults {
// public:
// 	SingleExecutionResults(int nExamples, int nSuccess, int nRejected) {
// 		this->nExamples = nExamples;
// 		this->nSuccess = nSuccess;
// 		this->nRejected = nRejected;
// 	}

// 	double successRate() { return ((double)nSuccess) / (nExamples - nRejected);}
// 	double rejectionRate() { return ((double)nRejected) / nExamples; }

// 	int nExamples;
// 	int nSuccess;
// 	int nRejected;
// };


// class KNNResults {
// public:
// 	SingleExecutionResults top1Result();
// 	SingleExecutionResults topXResult(int n);
// 	MatrixPointer getConfusionMatrix();
// 	MatrixPointer getPredictions();

	
// 	DatasetPointer getRawResults() {return results;}

// 	KNNResults(DatasetPointer results) { this->results = results;}

// private:
// 	DatasetPointer results;

// };


class KNNCUDA {
public:
	KNNCUDA (DatasetPointer train) { this->data = train; }

	KNNResults run(int k, DatasetPointer target, int b_s);
	// KNNResults runCuda(int k, DatasetPointer target);

private:
	DatasetPointer data;
};

#endif /* KNNCUDA_H */
