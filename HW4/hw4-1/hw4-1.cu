#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream> 
#include <cuda_runtime.h>
#include <chrono> 

using namespace std;

const int INF = ((1 << 30) - 1);
// const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
// void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;
// static int Dist[V][V];
static __align__(32) int *dist, *Result;

__global__ void PHASE_ONE(int Round, int B, int* distance, int V){
    // calculate target pivot
    extern __shared__ __align__(32) int shared_distance[];
    int x = threadIdx.x;
    int y = threadIdx.y;
    int p_x = Round * B + x; 
    int p_y = Round * B + y; 

    if(p_y < V && p_x < V){
        shared_distance[y * B + x] = distance[p_y * V + p_x];
    }else{
        shared_distance[y * B + x] = INF;
    }
    __syncthreads();

    #pragma unroll 
    for(int k = 0; k < B; k++){
        if(shared_distance[y * B + x] > shared_distance[y * B + k] + shared_distance[k * B + x]){
            shared_distance[y * B + x] = shared_distance[y * B + k] + shared_distance[k * B + x];
        }
        __syncthreads();
    }
    
    if(p_y < V && p_x < V){
        distance[p_y * V + p_x] = shared_distance[y * B + x];
    }

}
__global__ void PHASE_TWO(int Round, int B, int* distance, int V){
    // grid size (Round * 2)
    if(blockIdx.x == Round){
        return;
    }
    extern __shared__ __align__(32) int shared[];
    // first block stores pivot block
    // rest of spaces store distance col/row's block
    int* shared_p_distacne = shared; 
    int* shared_distance = shared + B*B;

    int x = threadIdx.x;
    int y = threadIdx.y;
    int p_x = Round * B + x;
    int p_y = Round * B + y;
    int idx_y, idx_x;

    // fill present distance
    if(p_y < V && p_x < V){
        shared_p_distacne[y * B + x] = distance[p_y * V + p_x];
    }else{
        shared_p_distacne[y * B + x] = INF;
    } 
    __syncthreads();

    if(blockIdx.x == Round) {
        return;
    }
  
    if(blockIdx.y == 0){
        // pivot row                    
        idx_x = blockIdx.x * B + x; 
        idx_y = p_y;
    }else{   
        // pivot col
        idx_x = p_x;
        idx_y = blockIdx.x * B + y;
    }

    if(idx_y >= V || idx_x >= V) {
        return;
    }

    if(idx_y < V && idx_x < V){
        shared_distance[y * B + x] = distance[idx_y * V + idx_x];
    }else{
        shared_distance[y * B + x] = INF;
    } 
    __syncthreads();

    // calculate for each row/col
    if(blockIdx.y == 0){        
        // #pragma unroll 
        for(int k = 0; k < B; k++){
            int temp = shared_p_distacne[y * B + k] + shared_distance[k * B + x];
            if(shared_distance[y * B + x] > temp){
                shared_distance[y * B + x] = temp;
            }  
            __syncthreads();
        }
    }else{            
        // #pragma unroll 
        for(int k = 0;  k < B; k++){
            int temp = shared_distance[y * B + k] + shared_p_distacne[k * B + x];
            if(shared_distance[y * B + x] > temp){
                shared_distance[y * B + x] = temp;
            }        
            __syncthreads();
        }     
    }

    if(idx_y < V && idx_x < V){
        distance[idx_y * V + idx_x] = shared_distance[y * B + x];
    }
    

}

__global__ void PHASE_THREE(int Round, int B, int* distance, int V){
    // grid size (Round*Round)
    if(blockIdx.x == Round || blockIdx.y == Round){
        return;
    }

    extern __shared__ __align__(32) int shared[];
    int* shared_row = shared;
    int* shared_col = shared + B*B;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int idx_x = blockIdx.x * B + x;
    int idx_y = blockIdx.y * B + y;
    int row_i = Round * B + y;
    int row_j = idx_x;
    int col_i = idx_y;
    int col_j = Round * B + x;

    if(row_i < V && row_j < V){
        shared_row[y * B + x] = distance[row_i * V + row_j];
    }else{
        shared_row[y * B + x] = INF;
    } 
    if(col_i < V && col_j < V){
        shared_col[y * B + x] = distance[col_i * V + col_j];
    }else{
        shared_col[y * B + x] = INF;
    }
    __syncthreads();

    if(idx_y >= V || idx_x >= V) {
        return;
    }

    int temp = distance[idx_y * V + idx_x];      
    // #pragma unroll 
    // #pragma GCC ivdep
    for(int k = 0; k < B; k++){
      if(temp > shared_col[y * B + k] + shared_row[k * B + x]){
        temp = shared_col[y * B + k] + shared_row[k * B + x];
      }
    }
    distance[idx_y * V + idx_x] = temp;
}

int main(int argc, char* argv[]) {
    input(argv[1]);
    int B = 32;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    dist = (int*)malloc(n * n * sizeof(int));
    Result = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                dist[i * n + j] = 0;
            } else {
                dist[i * n + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        dist[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if(Result[i * n + j] >= INF){ 
                Result[i * n + j] = INF;
              }
        }
        fwrite(&Result[i * n], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {

    int round = ceil(n, B);
    int *device_distance;
    int deviceID;
    cudaDeviceProp prop;
    dim3 grid_p1(1, 1);
    dim3 grid_p2(round, 2);
    dim3 grid_p3(round, round);
    dim3 blk(B, B);

    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&prop, deviceID);
    
    if (!prop.deviceOverlap){
		printf("!prop.deviceOverlap\n");
  	}
    
    cudaSetDevice(0);
    cudaMalloc(&device_distance, n * n * sizeof(int));
    cudaMemcpy(device_distance, dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < round; i++) {
        PHASE_ONE<<<grid_p1, blk, B * B * sizeof(int)>>>(i, B, device_distance, n);
  
        PHASE_TWO<<<grid_p2, blk, B * B * sizeof(int) * 2>>>(i, B, device_distance, n);
  
        PHASE_THREE<<<grid_p3, blk, B * B * sizeof(int) * 2>>>(i, B, device_distance, n);
    }

    cudaMemcpy(Result, device_distance, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_distance);

}

// void cal(
//     int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
//     int block_end_x = block_start_x + block_height;
//     int block_end_y = block_start_y + block_width;

//     for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
//         for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
//             // To calculate B*B elements in the block (b_i, b_j)
//             // For each block, it need to compute B times
//             for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
//                 // To calculate original index of elements in the block (b_i, b_j)
//                 // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
//                 int block_internal_start_x = b_i * B;
//                 int block_internal_end_x = (b_i + 1) * B;
//                 int block_internal_start_y = b_j * B;
//                 int block_internal_end_y = (b_j + 1) * B;

//                 if (block_internal_end_x > n) block_internal_end_x = n;
//                 if (block_internal_end_y > n) block_internal_end_y = n;

//                 for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
//                     for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
//                         if (Dist[i][k] + Dist[k][j] < Dist[i][j]) {
//                             Dist[i][j] = Dist[i][k] + Dist[k][j];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }