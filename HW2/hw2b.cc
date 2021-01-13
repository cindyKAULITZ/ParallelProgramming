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
    int x, y,p;
    png_bytep color;
    for (y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        // #pragma omp parallel for num_threads(40)
        for (x = 0; x < width; ++x) {
            p = buffer[(height - 1 - y) * width + x];
            color = row + x * 3;
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

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));

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
    const double y0_term = ((upper - lower) / height);
    const double x0_term = ((right - left) / width);

    int rank, size;
    int* image;
    int row_count ;
    // int availiable_// cpu += CPU_COUNT(&cpu_set);
    double time, t1,t2, comm2,comm1,cpu2,cpu1, cpu,comm;
    double r1,r2,r3,r0;
    double r1_cpu,r2_cpu,r3_cpu,r0_cpu;
	MPI_Status status, status1;

	// start parallel computation
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0){
        t1 = MPI_Wtime();

        /* allocate memory for image */
        image = (int*)malloc(width * height * sizeof(int));
        assert(image);

        row_count = 0;

        // comm1= MPI_Wtime();
        for (int c = 1; c < size; c++){
            MPI_Send(&row_count, 1, MPI_INT, c, 1, MPI_COMM_WORLD);
            row_count++;

        }
        // comm2= MPI_Wtime();
        // comm+= comm2-comm1;

        int free_rank = -1;
        int stop_proc = size-1;

        int* tmp = (int*)malloc(width * sizeof(int));
        assert(tmp);
        int write_j = 0;
        // int stop[size] = {0};

        do {
            // comm1= MPI_Wtime();

            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status1);
            free_rank = status1.MPI_SOURCE;
            write_j = status1.MPI_TAG;
            MPI_Recv(tmp, width, MPI_INT, MPI_ANY_SOURCE, write_j , MPI_COMM_WORLD, &status1);

            // comm2= MPI_Wtime();
            // comm+= comm2-comm1;
            // #pragma omp parallel num_threads(10)
            // {
                // #pragma omp for nowait
            // cpu1 = MPI_Wtime();

            #pragma GCC ivdep
            for(int c = 0; c<width;c++){
                image[write_j*width + c] = tmp[c];
            }
            // cpu2 = MPI_Wtime();
            // cpu += cpu2-cpu1;

            // }

            // comm1= MPI_Wtime();

            if( row_count < height){
                MPI_Send(&row_count, 1, MPI_INT, free_rank, 1, MPI_COMM_WORLD);
                row_count++;
            }else{
                // printf("send stop signal for rank %d, stop_proc = %d\n", free_rank,stop_proc);
                MPI_Send(&row_count, 1, MPI_INT, free_rank, 2, MPI_COMM_WORLD);
                stop_proc--;
                // stop[free_rank] = 1;
            }
            // comm2= MPI_Wtime();
            // comm+= comm2-comm1;
            
        }while(stop_proc > 0);

    }else{
        int j = 0;
        int stop = 0;
        int *sub_image;
        int repeats = 0; // iteration number
        double x = 0;    // real part of Z_repeat
        double y = 0;    // imag part of Z_repeat
        double length_squared = 0;
        double x0=0, y0=0;
        // cpu = 0;
        // comm= 0;

        // comm1= MPI_Wtime();

        MPI_Recv(&j, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        stop = status.MPI_TAG;

        // comm2= MPI_Wtime();
        // comm+= comm2-comm1;

        sub_image = (int*)malloc(width * sizeof(int));
        assert(sub_image);
        int n_t = 0;

        if(width > 1500){
            n_t = 80;
        }else{
            n_t = 50;
        }

        
        while (true){

            // cpu1 = MPI_Wtime();
            y0 = j * y0_term + lower;
            #pragma omp parallel num_threads(n_t)
            {   
                #pragma omp for schedule(guided) nowait
                for (int i = 0; i < width; ++i) {
                    x0 = i * x0_term + left;
                    repeats = 0; // iteration number
                    x = 0;    // real part of Z_repeat
                    y = 0;    // imag part of Z_repeat
                    length_squared = 0;
                    
                    while (repeats < iters && length_squared < 4) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    sub_image[i] = repeats;
                }
            }
            // cpu2 = MPI_Wtime();
            // cpu += cpu2-cpu1;

            // comm1= MPI_Wtime();

            MPI_Send(sub_image, width, MPI_INT,0 ,j ,MPI_COMM_WORLD);
            
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            stop = status.MPI_TAG;

            // comm2= MPI_Wtime();
            // comm+= comm2-comm1;

            if (stop == 1){
                // comm1= MPI_Wtime();

                MPI_Recv(&j, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

                // comm2= MPI_Wtime();
                // comm+= comm2-comm1;
            }else{
                // comm1= MPI_Wtime();

                MPI_Recv(&j, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);

                // comm2= MPI_Wtime();
                // comm+= comm2-comm1;

                free(sub_image);
                break;
                
            }


        }
        // printf("r%d // commtime = %.3f\n",rank, comm);
        // printf("r%d // cpu time = %.3f\n",rank, cpu);
        

    }

    // MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){

        // cpu1 = MPI_Wtime();

        write_png(filename, iters, width, height, image);
        free(image);

        // cpu2 = MPI_Wtime();
        // cpu += cpu2-cpu1;

        t2 = MPI_Wtime();
        // printf("cost time = %.3f\n", t2-t1);
        // printf("0 // commtime = %.3f\n", comm);
        // printf("0 // cpu time = %.3f\n", cpu);
    }
    MPI_Finalize();

}
