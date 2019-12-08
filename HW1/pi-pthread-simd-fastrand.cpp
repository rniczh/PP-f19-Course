#include <climits>
#include <cstdio>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>

#pragma GCC traget("sse4")

#define RMAX (32768.0 * 32768.0)

unsigned long long block = 0;

inline int fastrand(int *seed) {
  *seed = (214013 * *seed + 2531011);
  return (*seed >> 16) & 0x7FFF;
}

inline __m256 rand_simd(int *seeds) {
  return (__m256){(float)fastrand(&seeds[0]), (float)fastrand(&seeds[1]),
                  (float)fastrand(&seeds[2]), (float)fastrand(&seeds[3]),
                  (float)fastrand(&seeds[4]), (float)fastrand(&seeds[5]),
                  (float)fastrand(&seeds[6]), (float)fastrand(&seeds[7])};
}

inline __m256 simd_mult_add(__m256 a, __m256 b) {
  return (__m256)((__m256)(a * a) + (__m256)(b * b));
}

void *thread_circle(void *sum) {
  unsigned long long size = block;
  int cl = clock();
  int seeds[8] = { cl, cl + 1, cl + 2, cl + 3, cl + 4, cl + 5, cl + 6, cl + 7 };

  unsigned long long n = 0;
  register __m256 xmm_r0, ymm_r0, dmm0, xmm0, ymm0, zmm0;

  for (auto toss = 0; toss < size - 8; toss += 8) {
    xmm0 = rand_simd(seeds);
    ymm0 = rand_simd(seeds);
    zmm0 = simd_mult_add(xmm0, ymm0);

    n += ((zmm0[0] <= RMAX) + (zmm0[1] <= RMAX) + (zmm0[2] <= RMAX) +
          (zmm0[3] <= RMAX) + (zmm0[4] <= RMAX) + (zmm0[5] <= RMAX) +
          (zmm0[6] <= RMAX) + (zmm0[7] <= RMAX));
  }
  *(unsigned long long *)sum = n;
  pthread_exit(0);
}

int main(int argc, char **argv) {
  double pi_estimate, distance_squared, x, y;
  unsigned long long number_of_cpu, number_of_tosses, number_in_circle, toss;

  if (argc < 2) {
    exit(-1);
  }

  number_of_cpu = atoi(argv[1]);
  number_of_tosses = atoi(argv[2]);
  if ((number_of_cpu < 1) || (number_of_tosses < 0)) {
    exit(-1);
  }

  number_in_circle = 0;
  block = number_of_tosses / number_of_cpu;

  pthread_t *thread = (pthread_t *)malloc(number_of_cpu * sizeof(pthread_t));
  unsigned long long *sum =
      (unsigned long long *)calloc(number_of_cpu, sizeof(unsigned long long));
  for (auto i = 0; i < number_of_cpu; ++i) {
    pthread_create(&thread[i], 0, thread_circle, &sum[i]);
  }

  for (auto i = 0; i < number_of_cpu; ++i) {
    pthread_join(thread[i], 0);
  }

  number_in_circle = 0;
  for (auto i = 0; i < number_of_cpu; ++i) {
    number_in_circle += sum[i];
  }

  pi_estimate = 4 * number_in_circle / ((double)number_of_tosses);
  printf("%f\n", pi_estimate);

  free(thread);
  free(sum);
  return 0;
}
