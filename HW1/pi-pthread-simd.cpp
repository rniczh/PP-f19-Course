#include <iostream>
#include <pthread.h>
#include <immintrin.h>

#pragma GCC target("sse4")

unsigned long long block = 0;

void *thread_circle(void *sum) {
  unsigned long long size = block;
  unsigned int tid = pthread_self();

  unsigned int seeds[4] = {tid, tid + 1, tid + 2, tid + 3};
  unsigned long long n = 0;
  register __m256d dmm0, xmm_r0, ymm_r0, xmm0, ymm0, zmm0;

  // setup the RAND_MAX for later division
  dmm0 = (__m256d){RAND_MAX, RAND_MAX, RAND_MAX, RAND_MAX};


  for (auto toss = 0; toss < size - 4; toss += 4) {
    xmm_r0 = (__m256d){(double)rand_r(&seeds[0]), (double)rand_r(&seeds[1]),
                       (double)rand_r(&seeds[2]), (double)rand_r(&seeds[3])};
    ymm_r0 = (__m256d){(double)rand_r(&seeds[0]), (double)rand_r(&seeds[1]),
                       (double)rand_r(&seeds[2]), (double)rand_r(&seeds[3])};
    // vec_x
    xmm0 = (__m256d)(xmm_r0 / dmm0);
    // vec_y
    ymm0 = (__m256d)(ymm_r0 / dmm0);
    // vec_z = vec_x * vec_x + vec_y * vec_y
    zmm0 = (__m256d)((__m256d)(xmm0 * xmm0) + (__m256d)(ymm0 * ymm0));

    n += ((zmm0[0] <= 1) + (zmm0[1] <= 1) + (zmm0[2] <= 1) + (zmm0[3] <= 1));
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

  // pi_estimate = compute_pi_avx_unroll(number_of_tosses);

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
