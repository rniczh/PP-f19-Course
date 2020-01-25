/*************************************************************************
 * DESCRIPTION:
 *   Parallel Concurrent Wave Equation - C Version
 *   This program implements the concurrent wave equation by using CUDA
 ************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ceild(n, d) ceil(((double)(n)) / ((double)(d)))

#define THREADS 96

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

void check_param(void);
void update(void);
void printfinal(void);

int nsteps,                  /* number of time steps */
  tpoints,                 /* total points along string */
  rcode;                   /* generic return code */
float values[MAXPOINTS + 2], /* values at time t */
  oldval[MAXPOINTS + 2],   /* values at time (t-dt) */
  newval[MAXPOINTS + 2];   /* values at time (t+dt) */

/**********************************************************************
 *Checks input values from parameters
 *********************************************************************/
void check_param(void) {
  char tchar[20];

  /* check number of points, number of iterations */
  while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) {
    printf("Enter number of points along vibrating string [%d-%d]: ", MINPOINTS,
           MAXPOINTS);
    scanf("%s", tchar);
    tpoints = atoi(tchar);
    if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
      printf("Invalid. Please enter value between %d and %d\n", MINPOINTS,
             MAXPOINTS);
  }
  while ((nsteps < 1) || (nsteps > MAXSTEPS)) {
    printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
    scanf("%s", tchar);
    nsteps = atoi(tchar);
    if ((nsteps < 1) || (nsteps > MAXSTEPS))
      printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
  }

  printf("Using points = %d, steps = %d\n", tpoints, nsteps);
}

/**********************************************************************
 *      Calculate new values using wave equation
 *********************************************************************/
__device__ float do_math(float val, float old) {
  float dtime, c, dx, tau, sqtau;

  dtime = 0.3;
  c = 1.0;
  dx = 1.0;
  tau = (c * dtime / dx);
  sqtau = tau * tau;
  return (2.0 * val) - old + (sqtau * (-2.0) * val);
}

#define GET_INDEX(nblock) (1 + threadIdx.x + blockIdx.x * nblock)

/**********************************************************************
 *     Update all values along line a specified number of times
 *********************************************************************/
__global__ void update(float *device_values, int tpoints, int nsteps) {
  int i;
  float values1, newval1, oldval1;
  int idx = GET_INDEX(THREADS); /* k */

  if ((idx == 1) || (idx == tpoints)) {
    values1 = 0.0;
  } else {
    /* initialize this point */
    /* Calculate initial values based on sine curve */
    float x, fac, tmp;
    fac = 2.0 * PI;
    tmp = tpoints - 1;

    /* initialize this point */
    /* Calculate initial values based on sine curve */
    x = (float)(idx - 1)/tmp;
    values1 = sin(fac * x);
    oldval1 = values1;

    /* for each step */
    for (i = 1; i <= nsteps; ++i) {
      /* Update each point for this time step  */
      newval1 = do_math(values1, oldval1);
      oldval1 = values1;
      values1 = newval1;
    }
  }

  device_values[idx] = values1;
}
/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal() {
  int i;

  for (i = 1; i <= tpoints; i++) {
    printf("%6.4f ", values[i]);
    if (i % 10 == 0)
      printf("\n");
  }
}

/**********************************************************************
 *Main program
 *********************************************************************/
int main(int argc, char *argv[]) {
  sscanf(argv[1], "%d", &tpoints);
  sscanf(argv[2], "%d", &nsteps);
  check_param();

  /* setup cuda env */
  float *device_values;
  int size  = (1 + tpoints) * sizeof(float);
  int block = ceild(tpoints, THREADS);

  cudaMalloc((void **)&device_values, size);

  printf("Initializing points on the line...\n");
  printf("Updating all points for all time steps...\n");

  update<<<block, THREADS>>>(device_values, tpoints, nsteps);

  /* move result back to host */
  cudaMemcpy(values, device_values, size, cudaMemcpyDeviceToHost);

  printf("Printing final results...\n");
  printfinal();
  printf("\nDone.\n\n");

  return 0;
}
