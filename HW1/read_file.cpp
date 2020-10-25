#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main(int argc, char** argv) {
    FILE *file = fopen("./test_10.in", "wb"); 
    float test_case[10] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    int arrSize = sizeof(test_case)/sizeof(test_case[0]);

    for (int i=0; i< arrSize; ++i){
        fwrite(&test_case[i], sizeof(test_case[i]), 1, file);
    }
    return 0 ;
}