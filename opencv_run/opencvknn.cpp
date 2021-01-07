#include <fstream>
#include <string>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;
using namespace ml;

void read(string fileName, cv::Mat & data, cv::Mat & lebel){
    ifstream inputFile(fileName); 
    unsigned long long numSamples, numFeatures;
    inputFile >> numSamples >> numFeatures;
    cv::Mat rData(numSamples, numFeatures, CV_32F), rLabel(numSamples, 1, CV_32S);
    for(unsigned long long i = 0; i < numSamples;++i){
        for(unsigned long long j = 0; j < numFeatures; ++j){
            inputFile >> rData.at<float>(i, j);
        }
        inputFile >> rLabel.at<int>(i);
    }
    data = rData;
    lebel = rLabel;
}

Mat knn(const Mat & trainData, const Mat & trainLabel, const Mat & testData, const int k);
Mat MeanNormalize(Mat & data);
void ApplyMeanNormalization(Mat & data, const Mat & meanData);

double accuracy(const Mat & testLabel, const Mat & retLabel);

int main(int argc, char ** argv){
    if(argc != 4){
        printf("Should give 1: train data file name, 2: test data file name, 3: # k\n"
                "The train and test data format is same as knn project\n");
        return -1;
    }
    int k = stoi(argv[3]);
    cv::Mat trainData, trainLabel;
    cv::Mat testData, testLabel;
    read(argv[1], trainData, trainLabel);
    read(argv[2], testData, testLabel);
    Mat meanData = MeanNormalize(trainData);
    ApplyMeanNormalization(testData, meanData);
    Mat retLabel = knn(trainData, trainLabel, testData, k);

    printf("Accuracy: %.2lf% \n", accuracy(testLabel, retLabel));

    return 0;
}

Mat knn(const Mat & trainData, const Mat & trainLabel, const Mat & testData, const int k){
    Mat retLabel;
    Ptr<TrainData> train;
    Ptr<KNearest> knnClassifier = KNearest::create();
    train = TrainData::create(trainData, SampleTypes::ROW_SAMPLE, trainLabel);
    knnClassifier->setIsClassifier(true);
    knnClassifier->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
    knnClassifier->setDefaultK(k);
    knnClassifier->train(train);
    knnClassifier->findNearest(testData, knnClassifier->getDefaultK(), retLabel);
    return retLabel;
}

double accuracy(const Mat & testLabel, const Mat & retLabel){
    unsigned long long rows = testLabel.rows;
    unsigned long long correct = 0;
    for(unsigned long long i = 0; i < rows; ++i){
        if(testLabel.at<int>(i) == retLabel.at<int>(i)){
            correct += 1;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(rows) * 100.0;
}

Mat MeanNormalize(Mat & data) {
	/*
	 X =    X - X_min   /
	      X_max - X_min

	Returns a matrix: first row contains the "min" for each column. second row contains the "max" for each column
	*/
    Mat results = cv::Mat::zeros(2, data.cols, CV_32F);

	for(size_t i = 0; i < data.rows; i++) {
		for(size_t j = 0; j < data.cols; j++) {
			results.at<float>(0,j) = min(results.at<float>(0,j), data.at<float>(i,j));
			results.at<float>(1,j) = max(results.at<float>(1,j), data.at<float>(i,j));
		}
	}

	for(size_t i = 0; i < data.rows; i++) {
		for(size_t j = 0; j < data.cols; j++) {
			data.at<float>(i,j) = (data.at<float>(i,j) - results.at<float>(0,j)) / (results.at<float>(1,j) - results.at<float>(0,j));
		}
	}
	return results;
}

void ApplyMeanNormalization(Mat & data, const Mat & meanData) {
	for(size_t i = 0; i < data.rows; i++) {
			for(size_t j = 0; j < data.cols; j++) {
				data.at<float>(i,j) = (data.at<float>(i,j) - meanData.at<float>(0,j)) / (meanData.at<float>(1,j) - meanData.at<float>(0,j));
			}
		}
}
