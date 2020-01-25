#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include <sys/time.h>

#include <sstream>

#ifdef __APPLE__
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <OpenCL/opencl.h>
#else
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif

#ifdef DEBUG
#define START_TIME(start)                                                      \
  { gettimeofday(&start, NULL); }
#else
#define START_TIME(start)                                                      \
  {}
#endif

#ifdef DEBUG
#define END_TIME(name, end)                                                    \
  {                                                                            \
    gettimeofday(&end, NULL);                                                  \
    double delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec -     \
                    start.tv_usec) /                                           \
                   1.e6;                                                       \
    printf("\033[92m[%s]\033[0m Execute time: %lf\n", name, delta);            \
  }
#else
#define END_TIME(name, end)                                                    \
  {}
#endif

typedef struct {
  uint8_t R;
  uint8_t G;
  uint8_t B;
  uint8_t align;
} RGB;

typedef struct {
  bool type;
  uint32_t size;
  uint32_t height;
  uint32_t weight;
  RGB *data;
} Image;

Image *readbmp(const char *filename) {
  std::ifstream bmp(filename, std::ios::binary);
  char header[54];
  bmp.read(header, 54);
  uint32_t size = *(int *)&header[2];
  uint32_t offset = *(int *)&header[10];
  uint32_t w = *(int *)&header[18];
  uint32_t h = *(int *)&header[22];
  uint16_t depth = *(uint16_t *)&header[28];
  if (depth != 24 && depth != 32) {
    printf("we don't suppot depth with %d\n", depth);
    exit(0);
  }
  bmp.seekg(offset, bmp.beg);

  Image *ret = new Image();
  ret->type = 1;
  ret->height = h;
  ret->weight = w;
  ret->size = w * h;

  ret->data = new RGB[w * h]{};
  for (int i = 0; i < ret->size; i++) {
    bmp.read((char *)&ret->data[i], depth / 8);
  }

  return ret;
}

void writebmp(const char *filename, Image *img) {

  uint8_t header[54] = {
      0x42,          // identity : B
      0x4d,          // identity : M
      0,    0, 0, 0, // file size
      0,    0,       // reserved1
      0,    0,       // reserved2
      54,   0, 0, 0, // RGB data offset
      40,   0, 0, 0, // struct BITMAPINFOHEADER size
      0,    0, 0, 0, // bmp width
      0,    0, 0, 0, // bmp height
      1,    0,       // planes
      32,   0,       // bit per pixel
      0,    0, 0, 0, // compression
      0,    0, 0, 0, // data size
      0,    0, 0, 0, // h resolution
      0,    0, 0, 0, // v resolution
      0,    0, 0, 0, // used colors
      0,    0, 0, 0  // important colors
  };

  // file size
  uint32_t file_size = img->size * 4 + 54;
  header[2] = (unsigned char)(file_size & 0x000000ff);
  header[3] = (file_size >> 8) & 0x000000ff;
  header[4] = (file_size >> 16) & 0x000000ff;
  header[5] = (file_size >> 24) & 0x000000ff;

  // width
  uint32_t width = img->weight;
  header[18] = width & 0x000000ff;
  header[19] = (width >> 8) & 0x000000ff;
  header[20] = (width >> 16) & 0x000000ff;
  header[21] = (width >> 24) & 0x000000ff;

  // height
  uint32_t height = img->height;
  header[22] = height & 0x000000ff;
  header[23] = (height >> 8) & 0x000000ff;
  header[24] = (height >> 16) & 0x000000ff;
  header[25] = (height >> 24) & 0x000000ff;

  std::ofstream fout;
  fout.open(filename, std::ios::binary);
  fout.write((char *)header, 54);
  fout.write((char *)img->data, img->size * 4);
  fout.close();
}

cl_program load_program(cl_context context, std::vector<cl_device_id> &devices,
                        const char *filename) {
  std::ifstream in(filename, std::ios_base::binary);

  if (!in.good()) {
    return 0;
  }

  // get file length
  in.seekg(0, std::ios_base::end);
  size_t length = in.tellg();
  in.seekg(0, std::ios_base::beg);

  // read program source
  std::vector<char> data(length + 1);
  in.read(&data[0], length);
  data[length] = 0;

  // create and build program
  const char *source = &data[0];

  cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);

  if (program == 0) {
    return 0;
  }

  cl_int err;
  if ((err = clBuildProgram(program, 0, 0, 0, 0, 0)) != CL_SUCCESS) {
    printf("Error code: %d\n", err);
    size_t log_size;
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    // Allocate memory for the log
    char *log = (char *)malloc(log_size);
    // Get the log
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size,
                          log, NULL);
    // Print the log
    printf("%s\n", log);
    return 0;
  }
  return program;
}

int main(int argc, char *argv[]) {
  struct timeval start, end;

  cl_int err;
  cl_uint num;

  START_TIME(start);
  // =================================================================
  //                     ugly opencl API setup
  // =================================================================
  err = clGetPlatformIDs(0, 0, &num);
  std::vector<cl_platform_id> platforms(num);
  err = clGetPlatformIDs(num, &platforms[0], &num);

  cl_context_properties prop[] = {
      CL_CONTEXT_PLATFORM,
      reinterpret_cast<cl_context_properties>(platforms[0]), 0};

  cl_context context =
      clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);

  if (context == 0) {
    std::cerr << "Can't create OpenCL context\n";
    return 0;
  }

  size_t cb;
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
  std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
  clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);
  clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
  std::string devname;
  devname.resize(cb);
  clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
  std::cout << "Device: " << devname.c_str() << "\n";

  cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);

  if (queue == 0) {
    std::cerr << "Can't create command queue\n";
    clReleaseContext(context);
    return 0;
  }
  // =================================================================
  //            ugly opencl API end & setup the program
  // =================================================================
  // create buffer
  const unsigned int DATA_SIZE = 5000 * 5000;

  // Memory
  // cl_mem data_mem =
  //     clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
  //                    DATA_SIZE * sizeof(uint32_t), &img->data[0], 0);
  cl_mem R_mem =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, 256 * sizeof(uint32_t), 0, 0);
  cl_mem G_mem =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, 256 * sizeof(uint32_t), 0, 0);
  cl_mem B_mem =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, 256 * sizeof(uint32_t), 0, 0);

  // load program
  cl_program program = load_program(context, devices, "histogram.cl");
  if (program == 0) {
    // there exists some compile errors for cl file when this line be rised.
    std::cerr << "Can't load or build program\n";
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
  }

  // load kernel
  cl_kernel kernel_func = clCreateKernel(program, "histogram", 0);
  if (kernel_func == 0) {
    std::cerr << "Can't load kernel\n";
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
  }
  // =================================================================
  //                 endup the program configuration
  // =================================================================
  END_TIME("opencl configuration", end);

  char *filename;
  if (argc >= 2) {
    int many_img = argc - 1;
    for (int i = 0; i < many_img; i++) {
      filename = argv[i + 1];

      START_TIME(start);
      Image *img = readbmp(filename);
      END_TIME("readbmp to Image Object", end);

      std::cout << img->weight << ":" << img->height << "\n";

      uint32_t R[256];
      uint32_t G[256];
      uint32_t B[256];

      START_TIME(start);
      size_t work_size = img->weight * img->height;

      // create data memory buffer each times
      cl_mem data_mem = clCreateBuffer(
          context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          img->weight * img->height * sizeof(RGB), &img->data[0], 0);

      // setup the kernel arguments
      cl_uint init_value = 0;
      err = clEnqueueFillBuffer(queue, R_mem, &init_value, sizeof(cl_uint), 0,
                                256 * sizeof(cl_uint), 0, NULL, NULL);
      err = clEnqueueFillBuffer(queue, G_mem, &init_value, sizeof(cl_uint), 0,
                                256 * sizeof(cl_uint), 0, NULL, NULL);
      err = clEnqueueFillBuffer(queue, B_mem, &init_value, sizeof(cl_uint), 0,
                                256 * sizeof(cl_uint), 0, NULL, NULL);
      clSetKernelArg(kernel_func, 0, sizeof(cl_mem), &data_mem);
      clSetKernelArg(kernel_func, 1, sizeof(cl_mem), &R_mem);
      clSetKernelArg(kernel_func, 2, sizeof(cl_mem), &G_mem);
      clSetKernelArg(kernel_func, 3, sizeof(cl_mem), &B_mem);

      // execute the kernel function
      // histogram(img, R, G, B);
      err = clEnqueueNDRangeKernel(queue, kernel_func, 1, 0, &work_size, 0, 0,
                                   0, 0);

      if (err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, R_mem, CL_TRUE, 0,
                                  256 * sizeof(unsigned int), R, 0, 0, 0);
        err = clEnqueueReadBuffer(queue, G_mem, CL_TRUE, 0,
                                  sizeof(unsigned int) * 256, G, 0, 0, 0);
        err = clEnqueueReadBuffer(queue, B_mem, CL_TRUE, 0,
                                  sizeof(unsigned int) * 256, B, 0, 0, 0);
      } else {
        std::cout
            << "Error happen when execute the kernel function (Error Code): "
            << err << "\n";
      }

      // release memory buffer
      clReleaseMemObject(data_mem);

      END_TIME("historgram", end);

      START_TIME(start);

      int max = 0;
      for (int i = 0; i < 256; i++) {
        max = R[i] > max ? R[i] : max;
        max = G[i] > max ? G[i] : max;
        max = B[i] > max ? B[i] : max;
      }

      Image *ret = new Image();
      ret->type = 1;
      ret->height = 256;
      ret->weight = 256;
      ret->size = 256 * 256;
      ret->data = new RGB[256 * 256];

      for (int i = 0; i < ret->height; i++) {
        for (int j = 0; j < 256; j++) {
          ret->data[256 * i + j].R = R[j] * 256 / max > i ? 255 : 0;
          ret->data[256 * i + j].G = G[j] * 256 / max > i ? 255 : 0;
          ret->data[256 * i + j].B = B[j] * 256 / max > i ? 255 : 0;
        }
      }

      std::string newfile = "hist_" + std::string(filename);
      writebmp(newfile.c_str(), ret);
      END_TIME("write to bmp", end);
    }
  } else {
    printf("Usage: ./hist <img.bmp> [img2.bmp ...]\n");
  }
  return 0;
}
