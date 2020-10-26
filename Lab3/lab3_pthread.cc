#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	// cpu_set_t cpuset;
	// sched_getaffinity(0, sizeof(cpuset), &cpuset);
	// unsigned long long ncpus = CPU_COUNT(&cpuset);

	unsigned long long y, xx=0;
	unsigned long long base = r*r ;
	pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

	for (unsigned long long x = 0; x < r; x++) {
		// y = ceil(sqrtl(base));
		base = base - xx;
		pixels = pixels + ceil(sqrtl(base));
		// pthread_mutex_unlock(&mutex);
		// pthread_mutex_lock(&mutex);
		pthread_mutex_lock(&mutex);
		// base -= 2*x -1;
		xx = 2*x + 1;
		pthread_mutex_unlock(&mutex);
	}
	pixels %= k;
	// printf("pixels = %llu\n", pixels);

	printf("%llu\n", (4 * pixels) % k);
	pthread_mutex_destroy(&mutex);
}
