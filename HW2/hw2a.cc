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


int iters;
// [left, right]:  range of imag number
double left, right;
// [upper, lower]: range of real number 
double lower, upper;
int width, height;
// y0_term = ((upper - lower) / height)
// x0_term = ((right - left) / width)
double y0_term, x0_term;
size_t MAX_THREADS, MAX_QUEUE;
pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;

double get_y0(int row) { return row * y0_term + lower; }
double get_x0(int col) { return col * x0_term + left; }

bool next_pixel(int *p_row, int *p_col) {
    // assume (height > *p_row >= 0) and (width > *p_col >= 0) 
    if ( *p_col < width-1) {
        *p_col += 1;
        return true;      // remain in the same row
    } else if ( *p_col == width-1 ) {
        *p_col = 0;
        *p_row += 1;
        if (*p_row < height-1) {
            return true;  // end of the current row
        } else if (*p_row == height-1) {
            return false;  // end of image
        }
    }
    return false;
}

// bool check_iteration(int repeats_, double length_squared_) {
//     return (repeats_ < iters && length_squared_ < 4);
// }

struct pixel {
    int row_idx = 0;
    int col_idx = 0;
    int *image = NULL;
};

void task(void *arg) {
    pixel *p_pixel = (pixel*)arg;
    int j, i;
    // double x0, y0;
    pthread_mutex_lock(&task_mutex);
    i = p_pixel->col_idx;
    j = p_pixel->row_idx;
    next_pixel(&(p_pixel->row_idx), &(p_pixel->col_idx));
    pthread_mutex_unlock(&task_mutex);
    double x0 = get_x0(i);
    double y0 = get_y0(j);
    int repeats = 0; // iteration number
    double x = 0;    // real part of Z_repeat
    double y = 0;    // imag part of Z_repeat
    double length_squared = 0;
    while (repeats < iters && length_squared < 4) {
        double temp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = temp;
        length_squared = x * x + y * y;
        ++repeats;
    }
    p_pixel->image[j * width + i] = repeats;
}


// Thread pool
typedef enum {
    threadpool_invalid        = -1,
    threadpool_lock_failure   = -2,
    threadpool_queue_full     = -3,
    threadpool_shutdown       = -4,
    threadpool_thread_failure = -5
} threadpool_error_t;

typedef enum {
    threadpool_graceful       = 1
} threadpool_destroy_flags_t;


typedef enum {
    immediate_shutdown = 1,
    graceful_shutdown  = 2
} threadpool_shutdown_t;

/**
 *  @struct threadpool_task
 *  @brief the work struct
 *
 *  @var function Pointer to the function that will perform the task.
 *  @var argument Argument to be passed to the function.
 */

typedef struct {
    void (*function)(void *);
    void *argument;
} threadpool_task_t;

/**
 *  @struct threadpool
 *  @brief The threadpool struct
 *
 *  @var notify       Condition variable to notify worker threads.
 *  @var threads      Array containing worker threads ID.
 *  @var thread_count Number of threads
 *  @var queue        Array containing the task queue.
 *  @var queue_size   Size of the task queue.
 *  @var head         Index of the first element.
 *  @var tail         Index of the next element.
 *  @var count        Number of pending tasks
 *  @var shutdown     Flag indicating if the pool is shutting down
 *  @var started      Number of started threads
 */
struct threadpool_t {
  pthread_mutex_t lock;
  pthread_cond_t notify;
  pthread_t *threads;
  threadpool_task_t *queue;
  int thread_count;
  int queue_size;
  int head;
  int tail;
  int count;
  int shutdown;
  int started;
};

/**
 * @function void *threadpool_thread(void *threadpool)
 * @brief the worker thread
 * @param threadpool the pool which own the thread
 */

threadpool_t *threadpool_create(int thread_count, int queue_size, int flags);
int threadpool_add(threadpool_t *pool, void (*function)(void *), void *argument, int flags);
int threadpool_destroy(threadpool_t *pool, int flags);
int threadpool_free(threadpool_t *pool);
static void *threadpool_thread(void *threadpool);


int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    // fflush(stdout);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    // const char* filename = "pthread_test.png";

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
    y0_term = ((upper - lower) / height);
    x0_term = ((right - left)  /  width);


    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pixel pos;
    pos.image = image;
    MAX_THREADS = CPU_COUNT(&cpu_set) + 2;
    // MAX_THREADS = 1;
    MAX_QUEUE = width * height;
    int task_count = 0;

    if (MAX_THREADS == 1 ) {
        /* sequential verison of mandelbrot set */
        for (int j = 0; j < height; ++j) {
            double y0 = j * y0_term + lower;
            for (int i = 0; i < width; ++i) {
                double x0 = i * x0_term + left;
                int repeats = 0; // iteration number
                double x = 0;    // real part of Z_repeat
                double y = 0;    // imag part of Z_repeat
                double length_squared = 0;
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
    } else {
        // creat thread pool
        threadpool_t *pool = threadpool_create(MAX_THREADS, MAX_QUEUE, 0);
        // if (pool == NULL) {
        //     printf("threadpool_create() failed \n");
        //     fflush(stdout);
        //     return -1;
        // } else {
        //     printf("Pool started with %d threads and queue size of %d\n", MAX_THREADS, MAX_QUEUE);
        //     fflush(stdout);
        // }

        // send task into queue of thread pool
        for (size_t i = 0; i < MAX_QUEUE; i++ ) {
            threadpool_add(pool, &task, (void*)&pos, 0);
            // printf("task_count : %d\n", task_count++);
            // fflush(stdout);
        }
        // // send task into queue of thread pool
        // for (size_t i = 0; i < MAX_QUEUE; i++ ) {
        //     threadpool_add(pool, &write_task, (void*)&pos, 0);
        //     // printf("task_count : %d\n", task_count++);
        //     // fflush(stdout);
        // }
        // destory thread pool untill complete all tasks
        threadpool_destroy(pool, threadpool_graceful);
    }
    
    // /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}

threadpool_t *threadpool_create(int thread_count, int queue_size, int flags)
{
    threadpool_t *pool;
    int i;
    (void) flags;

    if(thread_count <= 0 || thread_count > MAX_THREADS || queue_size <= 0 || queue_size > MAX_QUEUE) {
        return NULL;
    }

    if((pool = (threadpool_t *)malloc(sizeof(threadpool_t))) == NULL) {
        goto err;
    }

    /* Initialize */
    pool->thread_count = 0;
    pool->queue_size = queue_size;
    pool->head = pool->tail = pool->count = 0;
    pool->shutdown = pool->started = 0;

    /* Allocate thread and task queue */
    pool->threads = (pthread_t *)malloc(sizeof(pthread_t) * thread_count);
    pool->queue = (threadpool_task_t *)malloc
        (sizeof(threadpool_task_t) * queue_size);

    /* Initialize mutex and conditional variable first */
    if((pthread_mutex_init(&(pool->lock), NULL) != 0) ||
       (pthread_cond_init(&(pool->notify), NULL) != 0) ||
       (pool->threads == NULL) ||
       (pool->queue == NULL)) {
        goto err;
    }

    /* Start worker threads */
    for(i = 0; i < thread_count; i++) {
        if(pthread_create(&(pool->threads[i]), NULL,
                          threadpool_thread, (void*)pool) != 0) {
            threadpool_destroy(pool, 0);
            return NULL;
        }
        pool->thread_count++;
        pool->started++;
    }

    return pool;

 err:
    if(pool) {
        threadpool_free(pool);
    }
    return NULL;
}

int threadpool_add(threadpool_t *pool, void (*function)(void *),
                   void *argument, int flags)
{
    int err = 0;
    int next;
    (void) flags;

    if(pool == NULL || function == NULL) {
        return threadpool_invalid;
    }

    if(pthread_mutex_lock(&(pool->lock)) != 0) {
        return threadpool_lock_failure;
    }

    next = (pool->tail + 1) % pool->queue_size;

    do {
        /* Are we full ? */
        if(pool->count == pool->queue_size) {
            err = threadpool_queue_full;
            break;
        }

        /* Are we shutting down ? */
        if(pool->shutdown) {
            err = threadpool_shutdown;
            break;
        }

        /* Add task to queue */
        pool->queue[pool->tail].function = function;
        pool->queue[pool->tail].argument = argument;
        pool->tail = next;
        pool->count += 1;

        /* pthread_cond_broadcast */
        if(pthread_cond_signal(&(pool->notify)) != 0) {
            err = threadpool_lock_failure;
            break;
        }
    } while(0);

    if(pthread_mutex_unlock(&pool->lock) != 0) {
        err = threadpool_lock_failure;
    }

    return err;
}

int threadpool_destroy(threadpool_t *pool, int flags)
{
    int i, err = 0;

    if(pool == NULL) {
        return threadpool_invalid;
    }

    if(pthread_mutex_lock(&(pool->lock)) != 0) {
        return threadpool_lock_failure;
    }

    do {
        /* Already shutting down */
        if(pool->shutdown) {
            err = threadpool_shutdown;
            break;
        }

        pool->shutdown = (flags & threadpool_graceful) ?
            graceful_shutdown : immediate_shutdown;

        /* Wake up all worker threads */
        if((pthread_cond_broadcast(&(pool->notify)) != 0) ||
           (pthread_mutex_unlock(&(pool->lock)) != 0)) {
            err = threadpool_lock_failure;
            break;
        }

        /* Join all worker thread */
        for(i = 0; i < pool->thread_count; i++) {
            if(pthread_join(pool->threads[i], NULL) != 0) {
                err = threadpool_thread_failure;
            }
        }
    } while(0);

    /* Only if everything went well do we deallocate the pool */
    if(!err) {
        threadpool_free(pool);
    }
    return err;
}

int threadpool_free(threadpool_t *pool)
{
    if(pool == NULL || pool->started > 0) {
        return -1;
    }
    /* Did we manage to allocate ? */
    if(pool->threads) {
        free(pool->threads);
        free(pool->queue);
        /* Because we allocate pool->threads after initializing the
           mutex and condition variable, we're sure they're
           initialized. Let's lock the mutex just in case. */
        pthread_mutex_lock(&(pool->lock));
        pthread_mutex_destroy(&(pool->lock));
        pthread_cond_destroy(&(pool->notify));
    }
    free(pool);    
    return 0;
}

static void *threadpool_thread(void *threadpool)
{
    threadpool_t *pool = (threadpool_t *)threadpool;
    threadpool_task_t task;

    for(;;) {
        /* Lock must be taken to wait on conditional variable */
        pthread_mutex_lock(&(pool->lock));

        /* Wait on condition variable, check for spurious wakeups.
           When returning from pthread_cond_wait(), we own the lock. */
        while((pool->count == 0) && (!pool->shutdown)) {
            pthread_cond_wait(&(pool->notify), &(pool->lock));
        }

        if((pool->shutdown == immediate_shutdown) ||
           ((pool->shutdown == graceful_shutdown) &&
            (pool->count == 0))) {
            break;
        }

        /* Grab our task */
        task.function = pool->queue[pool->head].function;
        task.argument = pool->queue[pool->head].argument;
        pool->head = (pool->head + 1) % pool->queue_size;
        pool->count -= 1;

        /* Unlock */
        pthread_mutex_unlock(&(pool->lock));

        /* Get to work */
        (*(task.function))(task.argument);
    }

    pool->started--;
    pthread_mutex_unlock(&(pool->lock));
    pthread_exit(NULL);
    return(NULL);
}