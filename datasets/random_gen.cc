#include <fstream>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <ctime>
using namespace std;

int main(int argc, char ** argv){
    if(argc != 5){
        printf("should have 4 arguments: 1. # samples, 2. # features, 3. #num classes, 4. output file name\n");
        return -1;
    }
    unsigned long long numSamples, numFeatures, numClasses;
    ofstream outputFile(argv[4]);
    numSamples = std::stoull(argv[1]);
    numFeatures = std::stoull(argv[2]);
    numClasses = std::stoull(argv[3]);
    srand((unsigned)time(nullptr));
    outputFile << numSamples << " " << numFeatures << endl;
    for(unsigned long long i = 0; i < numSamples; ++i){
        for(unsigned long long j = 0; j < numFeatures; ++j){
            float randNum = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 100; 
            outputFile << randNum << " ";
        }
        unsigned int c = rand() % numClasses;
        outputFile << c << endl;
    }
    return 0;
}
