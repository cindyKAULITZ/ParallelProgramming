#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

unsigned long long pixels = 0;
unsigned long long ncpus = 0;
unsigned long long r = 0;
unsigned long long x = 0;
unsigned long long xx = -1;
unsigned long long base = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* Compute(void* arg){
	for (; x < r; x++) {
		// y = ceil(sqrtl(base));
		// pthread_mutex_lock(&mutex);
		xx += 2;
		// pthread_mutex_unlock(&mutex);
		pixels = pixels + ceil(sqrtl(base));
		base = base - xx;
		
		// pthread_mutex_lock(&mutex);
		// base -= 2*x -1;
		// xx = (x*2) + 1;
		// pthread_mutex_unlock(&mutex);
	}
	pthread_exit(NULL);
}


int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	// unsigned long long pixels = 0;
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);
	pthread_t t[ncpus]; 
	// unsigned long long y, xx=0;
	base = r*r ;
	
	for (int i = 0 ; i <ncpus; i++){
		// write Compute function to calculate pixels
  		pthread_create(&t[i], NULL, Compute, NULL);
		pthread_join(t[i], NULL);
	}

	
	pixels %= k;
	// printf("pixels = %llu\n", pixels);

	printf("%llu\n", (4 * pixels) % k);
	pthread_mutex_destroy(&mutex);
}
