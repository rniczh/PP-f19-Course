#include <climits>
#include <cstdio>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <vector>

#pragma GCC target("sse4")

unsigned long long block = 0;

std::pair<__v8su, __v8su> fastrand(__v8su seed) {
  register __v8su multmm0, addmm0, maskmm0, sfmm0, mulres, addres, shiftres, result;
  multmm0 = (__v8su){214013, 214013, 214013, 214013,
                     214013, 214013, 214013, 214013};
  addmm0  = (__v8su){2531011, 2531011, 2531011, 2531011,
                     2531011, 2531011, 2531011, 2531011};
  maskmm0 = (__v8su){0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF,
                     0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF};
  sfmm0 = (__v8su){ 16, 16, 16, 16, 16, 16, 16, 16 };
  mulres = (__v8su){ multmm0 * seed };
  addres = (__v8su){ mulres + addmm0 };
  shiftres = (__v8su){ addres >> sfmm0 };

  result = (__v8su){ shiftres & maskmm0 };
  return std::make_pair(result, addres);
}

#define RS 32768 * 32768

void *thread_circle(void *sum) {
  unsigned long long size = block;
  unsigned int cl = clock();

  __v8su seeds = (__v8su){cl,     cl + 1, cl + 2, cl + 3,
                          cl + 4, cl + 5, cl + 6, cl + 7};

  unsigned long long n = 0;
  register __v8su xmm_r0, ymm_r0, dmm0, xmm0, ymm0, zmm0;

  // each loop sends 8 toss by using simd
  for (auto toss = 0; toss < size - 8; toss += 8) {
    auto xpair = fastrand(seeds);
    auto ypair = fastrand(xpair.second);
    seeds = ypair.second;

    xmm0 = xpair.first;
    ymm0 = ypair.first;

    zmm0 = (__v8su)((__v8su)(xmm0 * xmm0) + (__v8su)(ymm0 * ymm0));

    // compare the 8 result with RS
    // from:
    //    (x/r)^2 + (y/r)^2 <= 1
    // to:
    //    x^2 + y^2 <= r^2
    n += ((zmm0[0] <= RS) + (zmm0[1] <= RS) + (zmm0[2] <= RS) + (zmm0[3] <= RS) +
          (zmm0[4] <= RS) + (zmm0[5] <= RS) + (zmm0[6] <= RS) + (zmm0[7] <= RS));
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
