typedef struct {
  unsigned char R;
  unsigned char G;
  unsigned char B;
  unsigned char align;
} RGB;

/* void histogram(Image *img, uint32_t R[256], uint32_t G[256], uint32_t B[256])
 * { */
/*   std::fill(R, R + 256, 0); */
/*   std::fill(G, G + 256, 0); */
/*   std::fill(B, B + 256, 0); */

/*   for (int i = 0; i < img->size; i++) { */
/*     RGB &pixel = img->data[i]; */
/*     R[pixel.R]++; */
/*     G[pixel.G]++; */
/*     B[pixel.B]++; */
/*   } */
/* } */

__kernel void histogram(__global RGB *data,
                        __global unsigned int R[256],
                        __global unsigned int G[256],
                        __global unsigned int B[256]) {

  unsigned int idx = get_global_id(0);

  atomic_add(&R[data[idx].R], 1);
  atomic_add(&G[data[idx].G], 1);
  atomic_add(&B[data[idx].B], 1);
}