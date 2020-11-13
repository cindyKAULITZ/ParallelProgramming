#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}
// void write_png(const char* filename, int iters, int width, int height, const int* buffer, int row_count) {
//     FILE* fp;
//     png_structp png_ptr;
//     png_infop info_ptr;
//     static int tag = 0;
//     if (tag == 0){
//         fp = fopen(filename, "wb");
//         assert(fp);
//         png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
//         assert(png_ptr);
//         info_ptr = png_create_info_struct(png_ptr);
//         assert(info_ptr);
//         png_init_io(png_ptr, fp);
//         png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
//                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
//         png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
//         png_write_info(png_ptr, info_ptr);
//         png_set_compression_level(png_ptr, 1);
//         tag++;
//     }else{

//         size_t row_size = 3 * width * sizeof(png_byte);
//         png_bytep row = (png_bytep)malloc(row_size);
//         int y = row_count;

//         memset(row, 0, row_size);
//         for (int x = 0; x < width; ++x) {
//             int p = buffer[(height - 1 - y) * width + x];
//             png_bytep color = row + x * 3;
//             if (p != iters) {
//                 if (p & 16) {
//                     color[0] = 240;
//                     color[1] = color[2] = p % 16 * 16;
//                 } else {
//                     color[0] = p % 16 * 16;
//                 }
//             }
//         }
//         png_write_row(png_ptr, row);
        
//         free(row);
//         tag++;
//     }
//     if(tag == height-1){
//         png_write_end(png_ptr, NULL);
//         png_destroy_write_struct(&png_ptr, &info_ptr);
//         fclose(fp);

//     }
// }

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    // maximum iteration
    int iters = strtol(argv[2], 0, 10);
    // [left, right]:  range of imag number
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    // [upper, lower]: range of real number 
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    int rank, size;
    int free_proc;
    int* image;
    // int* sub_image;
    int row_count , task_count;
    // int tag = 0;

	MPI_Request request, request1 ;
	MPI_Status status;

	// start parallel computation
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("rank %d of %d\n", rank, size);
    
    if (rank == 0){
        /* allocate memory for image */
        image = (int*)malloc(width * height * sizeof(int));
        assert(image);
        // *sub_image = (int*)malloc(width * sizeof(int));
        // assert(image);
        // sub_image = (int*)malloc(width * sizeof(int));
        // assert(sub_image);
        
        // int free[size] = {0};
        int total_row = height;
        row_count = 0;
        task_count = 0;

        // free_proc = 1;
        /* create work pool */
        // data = 1, result = 2, terminate = -1
        printf("here\n");
        for (int c = 1; c < size; c++){
            printf("here c = %d\n", c);
            MPI_Isend(&row_count, 1, MPI_INT, c, 1, MPI_COMM_WORLD, &request);
            task_count++;
            row_count++;
            // free[c] = 1;

        }
        int free_rank = -1;
        // int j;
        int* tmp = (int*)malloc((width+5) * sizeof(int));
        assert(tmp);
        int tmp_j = 0;
        do {
            printf("in do\n");
            // MPI_Recv(&image, width, MPI_INT, MPI_ANY, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // MPI_Recv(&free_rank, 1, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("in do1\n");
            // MPI_Recv(&j, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("in do2\n");
            printf("in do2-1 %d\n", tmp[width]);
            int recv_len = width+5;
            MPI_Irecv(&tmp, recv_len, MPI_INT, MPI_ANY_SOURCE,2 , MPI_COMM_WORLD, &request1);
            // MPI_Wait(&request1, &status);
            int flag;
            MPI_Test(&request1, &flag, &status);
            if (flag != 0) { 
                printf("recv : , slave : %d\n", status.MPI_SOURCE);
            }
            printf("in do2-2 %d\n", tmp[width]);
            free_rank = tmp[width];
            printf("in do3 free_rank = %d\n", free_rank);
            int t_len = width+1;
            tmp_j = (int)tmp[t_len];
            printf("in do4 tmp_j = %d\n", tmp_j);
            for(int c = 0; c<width;c++){
                image[tmp_j*width + c] = tmp[c];
            }
            // printf("in do3\n");
            task_count--;
            // MPI_Gather(&free_proc, 1, MPI_INT, &free, size, MPI_INT, 0, MPI_COMM_WORLD);

            if(free_rank != -1 && row_count < total_row){
                printf("in if %d, to rank %d\n", row_count, free_rank);
                MPI_Isend(&row_count, 1, MPI_INT, free_rank, 1, MPI_COMM_WORLD, &request);
                task_count++;
                row_count++;
                free_rank = -1;
                // free[c] = 1;
            }
            // for(int c = 1; c<size; c++){
            //     if(free[c] == 0 && row_count < total_row){
            //         MPI_Send(&row_count, 1, MPI_INT, c, row_count, MPI_COMM_WORLD);
            //         task_count++;
            //         row_count++;
            //         free[c] = 1;
            //     }
            // }
            printf("task_count= %d\n",task_count);
            
        }while(task_count > 0);

        write_png(filename, iters, width, height, image);
        free(image);



    }else{
        int j = 0;
        int *sub_image;
        printf("in else rank %d\n", rank);
        MPI_Irecv(&j, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
        // free_proc = 1;
        int recv_len = width+5;
        sub_image = (int*)malloc(recv_len * sizeof(int));
        assert(sub_image);

        while (j < height){
            printf("in outer while rank %d, j is %d of %d\n", rank, j, height);
            /* mandelbrot set */
            // int j = row_count;
            double y0 = j * ((upper - lower) / height) + lower;
            for (int i = 0; i < width; ++i) {
                double x0 = i * ((right - left) / width) + left;
                int repeats = 0; // iteration number
                double x = 0;    // real part of Z_repeat
                double y = 0;    // imag part of Z_repeat
                double length_squared = 0;
                // printf("inner while rank %d\n", rank);
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                // printf("end inner while rank %d\n", rank);
                sub_image[i] = repeats;
                // printf("write rank %d\n", rank);
            }
            sub_image[width] = (int)rank;
            sub_image[(int)(width+1)] = (int)j;
            sub_image[(int)(width+2)] = j;
            sub_image[(int)(width+3)] = j;
            sub_image[(int)(width+4)] = j;
            printf("end for loop rank %d, j = %d\n", sub_image[width], sub_image[width+1]);

            // MPI_Send(&rank, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            // MPI_Send(&j, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
            
            MPI_Isend(&sub_image, recv_len, MPI_INT,0 ,2 ,MPI_COMM_WORLD ,&request1);
            // free(sub_image);
            printf("To Recv new rank %d\n", rank);

            MPI_Irecv(&j, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, &status);
            printf("Recv new j %d\n", j);

        }


    }
    MPI_Finalize();



}
