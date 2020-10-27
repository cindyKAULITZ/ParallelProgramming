#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

unsigned long long pixels = 0;
unsigned long long ncpus = 0;
unsigned long long r = 0;
unsigned long long k = 0;
unsigned long long base = 0;
unsigned long long *sub_start;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* Compute(void* sub){
	unsigned long long* start = (unsigned long long*) sub;
	unsigned long long* end = (unsigned long long*) sub+1;
	// printf("start %d\tend%d\nr = %d\t base = %d\n", *start, *end,r ,base);
	unsigned long long pixel = 0;
	#pragma GCC ivdep
	for (unsigned long long x = *start; x < *end; x++) {

		// pixels = pixels + ceil(sqrtl(base));
		// pthread_mutex_lock(&mutex);
		// base = base - 2*x -1;
		// pthread_mutex_unlock(&mutex);
		
		pixel += ceil(sqrtl(base-x*x));
		// y += ceil(sqrtl(base-x*x));
	}
	pixel %= k;
	// printf()
	pthread_mutex_lock(&mutex);
	pixels += pixel;
	pthread_mutex_unlock(&mutex);

	pthread_exit(NULL);
}


int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	// printf("???");
	r = atoll(argv[1]);
	k = atoll(argv[2]);
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
	// printf("???");
	for (int i = 0 ; i <ncpus; i++){
		// write Compute function to calculate pixels
		sub_start[2*i] = (unsigned long long)i * batch;
		sub_start[(2*i)+1] = i * batch + batch;
		if (sub_start[(2*i)+1] > r) { sub_start[(2*i)+1] = r;}
		// sub_start[i] = i*;
  		pthread_create(&t[i], NULL, Compute, (void*) &sub_start[2*i]);
	}

	for (int i = 0; i<ncpus; i++) {
		pthread_join(t[i], NULL);
	}

	pixels %= k;
	// printf("pixels = %llu\n", pixels);

	printf("%llu\n", (4 * pixels) % k);
	pthread_mutex_destroy(&mutex);
}
