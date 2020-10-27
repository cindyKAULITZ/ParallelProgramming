#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

<<<<<<< Updated upstream
unsigned long long pixels = 0;
unsigned long long ncpus = 0;
unsigned long long r = 0;
unsigned long long x = 0;
unsigned long long xx = -1;
unsigned long long base = 0;
unsigned long long *sub_start;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* Compute(void* sub){
	unsigned long long* start = (unsigned long long*) sub;
	unsigned long long* end = (unsigned long long*) sub+1;
	// printf("start %d\tend%d\nr = %d\t base = %d\n", *start, *end,r ,base);
	unsigned long long y = 0;
	for (unsigned long long x = *start; x < *end; x++) {

		// pixels = pixels + ceil(sqrtl(base));
		// pthread_mutex_lock(&mutex);
		// base = base - 2*x -1;
		// pthread_mutex_unlock(&mutex);
		
		pthread_mutex_lock(&mutex);
		pixels += ceil(sqrtl(base-x*x));
		// y += ceil(sqrtl(base-x*x));
		pthread_mutex_unlock(&mutex);
	}
	// printf()
	// pixels += y;
=======

#define METHOD_1
// #define METHOD_2

unsigned long long r_square;
unsigned long long r;
unsigned long long k;
unsigned long long batch;
unsigned long long pixels = 0;

#ifdef METHOD_1
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
#endif


typedef struct data {
    unsigned long long thread_id;
    unsigned long long pixel = 0;
} data;


void* sub_task(void *arg) {
	data *pdata = (data*)arg;
	unsigned long long sub_start = pdata->thread_id * batch;
	unsigned long long sub_end = sub_start + batch;
	if (sub_end > r) { sub_end = r;}

	unsigned long long base = r_square - (sub_start*sub_start);
	unsigned long long xx = 0;
	unsigned long long sub_pixel = 0;
	for(unsigned long long ele_count = sub_start; ele_count < sub_end; ele_count++){
		base = base - xx;
		sub_pixel += ceil(sqrtl(base));
		xx = 2*ele_count + 1;		
	}
	sub_pixel %= k;
	pdata->pixel = sub_pixel;

#ifdef METHOD_1
	pthread_mutex_lock(&mutex);
	pixels += pdata->pixel;
	pthread_mutex_unlock(&mutex);
#endif
>>>>>>> Stashed changes
	pthread_exit(NULL);
}


int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
<<<<<<< Updated upstream
	unsigned long long k = atoll(argv[2]);
	// unsigned long long pixels = 0;
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);
	pthread_t t[ncpus]; 
	sub_start = (unsigned long long*)malloc(sizeof(unsigned long long)* ncpus * 2);
	base = r*r ;
	const unsigned long long batch = ((r-1)/(ncpus))+1;
	
	//caculate partial sum
	// unsigned long long sub_end = sub_start + batch;

	for (int i = 0 ; i <ncpus; i++){
		// write Compute function to calculate pixels
		sub_start[2*i] = (unsigned long long)i * batch;
		sub_start[(2*i)+1] = i * batch + batch;
		if (sub_start[(2*i)+1] > r) { sub_start[(2*i)+1] = r;}
		// sub_start[i] = i*;
  		pthread_create(&t[i], NULL, Compute, (void*) &sub_start[2*i]);
		pthread_join(t[i], NULL);
	}

	
	pixels %= k;
=======
	k = atoll(argv[2]);

    // acquire cpu number
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
    
	// compute batch
	batch = ((r-1)/(ncpus))+1;
	r_square = r*r;

    pthread_t threads[ncpus];
	data datas[ncpus];

	for (int i = 0 ; i <ncpus; i++){
		// part 0 is done by main thread
		datas[i].thread_id = i;
  		pthread_create(&threads[i], NULL, sub_task, (void*) &datas[i]);
	}

	for (int i = 0 ; i <ncpus; i++){
		pthread_join(threads[i], NULL);
	}

#ifdef METHOD_2
	for (int i = 0 ; i <ncpus; i++){
		pixels += datas[i].pixel;
	}
#endif

>>>>>>> Stashed changes
	// printf("pixels = %llu\n", pixels);
	printf("%llu\n", (4 * (pixels)) % k);

#ifdef METHOD_1
	pthread_mutex_destroy(&mutex);
#endif

}


