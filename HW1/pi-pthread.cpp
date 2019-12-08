#include <iostream>
#include <pthread.h>

unsigned long long block = 0;

void *thread_circle(void *tid) {
  unsigned long long size = block;
  unsigned int seed = clock();
  unsigned long long n = 0;
  for (auto toss = 0; toss < size; toss++) {
    double x = (double)rand_r(&seed)/RAND_MAX;
    double y = (double)rand_r(&seed)/RAND_MAX;
    if (x * x + y * y <= 1)
      n++;
  }

  unsigned long long *res = (unsigned long long*)malloc(sizeof(unsigned long long));
  *res = n;
  pthread_exit((void *)res);
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
  unsigned long long *sum = (unsigned long long*)calloc(number_of_cpu, sizeof(unsigned long long));
  for (auto i = 0; i < number_of_cpu; ++i) {
    pthread_create(&thread[i], 0, thread_circle, &i);
  }

  for (auto i = 0; i < number_of_cpu; ++i) {
    void *status;
    pthread_join(thread[i], &status);
    sum[i] = *(unsigned long long*)status;
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
