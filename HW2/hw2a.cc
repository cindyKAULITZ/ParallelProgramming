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
#include <pthread.h>
#include <chrono>
#include <time.h>

// maximum iteration
int iters;
// [left, right]:  range of imag number
double left;
double right;
// [upper, lower]: range of real number 
double lower;
double upper;
int width;
int height;
double x0_term, y0_term;
int j_count = 0;

/* allocate memory for image */
int* image;
double total=0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;



void *mandelbrot_set(void *id){
    int* tid = (int*) id;
    // int* start = (int*) sub;
	// int* end = (int*) sub+1;
    
    /* mandelbrot set */
    while(j_count < height) {
        pthread_mutex_lock(&mutex);
        int j = j_count;
        j_count++;
        pthread_mutex_unlock(&mutex);

        double y0 = j * y0_term + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * x0_term + left;
            int repeats = 0; // iteration number
            double x = 0;    // real part of Z_repeat
            double y = 0;    // imag part of Z_repeat
            double length_squared = 0;
            #pragma GCC ivdep
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
    }
    
	pthread_exit(NULL);
}

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

int main(int argc, char** argv) {
    auto start = std::chrono::system_clock::now();
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    // maximum iteration
    iters = strtol(argv[2], 0, 10);
    // [left, right]:  range of imag number
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    // [upper, lower]: range of real number 
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    y0_term = ((upper - lower) / height);
    x0_term = ((right - left) / width);

    /* create threads */
    // int ncpus = CPU_COUNT(&cpu_set);
    int ncpus = 15;
    int *id = (int*)malloc(ncpus*sizeof(int));
    // total = (double*)malloc(ncpus*sizeof(double));
	pthread_t t[ncpus]; 
    
    for (int i = 0 ; i <ncpus; i++){
        id[i] = i;
  		pthread_create(&t[i], NULL, mandelbrot_set, (void*)&id[i]);
	}

	for (int i = 0; i<ncpus; i++) {
		pthread_join(t[i], NULL);
	}


    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);

    pthread_mutex_destroy(&mutex);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    // printf("total cpu time:%f\n", elapsed_seconds);
}
