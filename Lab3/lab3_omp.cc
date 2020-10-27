#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long x = 0;
	// int threads = omp_get_num_threads();
	unsigned long long r_square = r*r;

#pragma omp parallel reduction(+:pixels)
{
	
	#pragma omp for nowait
	for (x = 0; x < r; x++) {
		// unsigned long long y = 
		pixels += ceil(sqrtl(r_square - x*x));
	}
	pixels %= k;
	// pixels+=pixel;
}
	printf("%llu\n", (4 * pixels) % k);
}
