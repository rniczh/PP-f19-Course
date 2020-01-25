#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef W
#define W 20 // Width
#endif
int main(int argc, char **argv) {
  /* dealing with the arguments */
  int L = atoi(argv[1]);         // Length
  int iteration = atoi(argv[2]); // Iteration
  srand(atoi(argv[3]));          // Seed

  /* setup the mpi enviorment */
  int rank, np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* the rank */
  MPI_Comm_size(MPI_COMM_WORLD, &np);   /* num of processor */


  MPI_Status stat;

  float d = (float)random() / RAND_MAX * 0.2; // Diffusivity
  int *temp = malloc(L * W * sizeof(int));    // Current temperature
  int *next = malloc(L * W * sizeof(int));    // Next time step

  /* Split */
  int avg_chunk_length = L / np;
  int cur_chunk_length =
      rank < L % np ? avg_chunk_length + 1 : avg_chunk_length;
  int from = rank <= L % np ?
    rank * (avg_chunk_length + 1) :
    (rank - L % np) * avg_chunk_length + (L % np) * (avg_chunk_length + 1);
  int to = from + cur_chunk_length - 1;

  /* Checking wheather it's first one or last one */
  const int first_one = rank == 0;
  const int last_one  = rank == np - 1;

  for (int i = 0; i < L; i++) {
    for (int j = 0; j < W; j++) {
      temp[i * W + j] = random() >> 3;
    }
  }

  int count = 0;

  int *balances = calloc(np, sizeof(*balances));
  int *global_balances = calloc(np, sizeof(*global_balances));

  while (iteration--) { // Compute with up, left, right, down points
    balances[rank] = 1;
    count++;

    for (int i = from; i <= to; i++) {
      for (int j = 0; j < W; j++) {
        float t = temp[i * W + j] / d;
        t += temp[i * W + j] * -4;
        t += temp[(i - 1 < from && first_one ? from : i - 1) * W + j];
        t += temp[(i + 1 > to && last_one ? to : i + 1) * W + j];
        t += temp[i * W + (j - 1 < 0 ? 0 : j - 1)];
        t += temp[i * W + (j + 1 >= W ? j : j + 1)];
        t *= d;
        next[i * W + j] = t;
        if (next[i * W + j] != temp[i * W + j]) {
          balances[rank] = 0;
        }
      }
    }

    int *tmp = temp;
    temp = next;
    next = tmp;

    /* Sync the boundary of the block here */
    if (!first_one) {
      /* send to left */
      MPI_Send(temp + from * W, W, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
    }

    if (!last_one) {
      /* recv right and send right */
      MPI_Recv(temp + (to + 1) * W, W, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &stat);
      MPI_Send(temp + to * W, W, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (!first_one) {
      /* recv left */
      MPI_Recv(temp + (from - 1) * W, 1 * W, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &stat);
    }

    /* Sync the balance */
    MPI_Allreduce(balances, global_balances, np, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
    int k = 0;
    for (int i = 0; i < np; ++i) {
      k += global_balances[i];
    }
    if (k == np)
      break;

  } /* until it's balanced */

  int min = temp[from * W];
  for (int i = from; i <= to; i++) {
    for (int j = 0; j < W; j++) {
      if (temp[i * W + j] < min) {
        min = temp[i * W + j];
      }
    }
  }

  int global_min = 0;
  MPI_Reduce(&min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Size: %d*%d, Iteration: %d, Min Temp: %d\n", L, W, count,
           global_min);
  }

  MPI_Finalize();

  return 0;
}
