#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <chrono>
using namespace std;

const int INF = ((1 << 30) - 1);
const int MAX_V = 6000;
int V, E;
int Dist[MAX_V][MAX_V];
void input(char* inFileName);
void output(char* outFileName);
void APSP(){
    
    for (int k = 0; k < V; k++) {
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
          Dist[i][j] = min( Dist[i][j], Dist[i][k] + Dist[k][j] );
        }
      }
    }
}


int main(int argc, char* argv[]) {
    // std::chrono::steady_clock::time_point r0 = std::chrono::steady_clock::now();
    input(argv[1]);
    // std::chrono::steady_clock::time_point r1 = std::chrono::steady_clock::now();
    // std::chrono::duration<double> read_t = r1 - r0;

    // std::chrono::steady_clock::time_point c0 = std::chrono::steady_clock::now();
    APSP();
    // std::chrono::steady_clock::time_point c1 = std::chrono::steady_clock::now();
    // std::chrono::duration<double> cal_t = c1 - c0;
    
    // std::chrono::steady_clock::time_point w0 = std::chrono::steady_clock::now();
    output(argv[2]);
    // std::chrono::steady_clock::time_point w1 = std::chrono::steady_clock::now();
    // std::chrono::duration<double> write_t = w1 - w0;
    // printf("Read File took %g\n", read_t.count());
    // printf("APSP File took %g\n", cal_t.count());
    // printf("Write File took %g\n", write_t.count());
    // printf("Read+Write File took %g\n", read_t.count()+write_t.count());
    return 0;
}

void input(char* infile) {
    
    FILE* file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), V, outfile);
    }
    fclose(outfile);
}