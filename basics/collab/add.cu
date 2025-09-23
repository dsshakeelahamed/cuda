%%writefile vector_add_1_0.cu
#include <iostream>
#include <cuda_runtime.h>

#define ERR_CHECK(err) \
  if (err != cudaSuccess) { \
    std::cout << "Error with cuda operation : " << cudaGetErrorString(err) << " at line " << __LINE__ <<std::endl; \
    return; \
  } \


__global__ void addVector(int* a, int* b, int* c) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  c[id] = a[id] + b[id];
  return;
}

__global__ void addVector2D(float* A, size_t pitch_A, float* B, size_t pitch_B, float* C, size_t pitch_C, int width, int height) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

   float* r_A = (float*)((char*)A + row * pitch_A);
   float* r_B = (float*)((char*)B + row * pitch_B);
   float* r_C = (float*)((char*)C + row * pitch_C);

  if (col < width && row < height) {
    r_C[col] = r_A[col] + r_B[col];
  }
  return;

}
void deviceConfig(){
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
  std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
  std::cout << "  Max block dimensions (X, Y, Z): ("
            << deviceProp.maxThreadsDim[0] << ", "
            << deviceProp.maxThreadsDim[1] << ", "
            << deviceProp.maxThreadsDim[2] << ")" << std::endl;
  std::cout << std::endl;
}

void oneDOperations() {
  int N = 1 << 11;
  size_t size = N * sizeof(int);

  int *a = new int[N];
  int *b = new int[N];
  int *c = new int[N];

  deviceConfig();

  // generate data
  for(int i=0; i < N ; i++) {
    a[i] = i;
    b[i] = i;
  }
  // set context
  cudaSetDevice(0);

  // create streams
  cudaStream_t stream;
  cudaStreamCreate(&stream);


  // allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

 // create events

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);

  // copy from host to device
  cudaMemcpyAsync(d_a, a, size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_b, b, size, cudaMemcpyHostToDevice, stream);

  //cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  // run kernel
  int threadsPerBlock = 1024;
  int blockSize = N / threadsPerBlock;
  addVector<<<blockSize, threadsPerBlock, 0, stream>>>(d_a,d_b,d_c);

  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  std::cout<<"Kernel took "<< ms << " ms\n" << std::endl ;

  // copy results
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);


  for(int i =0 ; i < 10; i++) {
    std::cout << "Index " << i << ", Value " << c[i] << std::endl;
  }

  delete[] a;
  delete[] b;
  delete[] c;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);

}


void twoDOperations() {
  size_t width = 1 << 8;
  size_t height = 1 << 8;
  //int N = width * height;
  //size_t size = N * sizeof(float);

  float *h_A = new float[height * width];
  float *h_B = new float[height * width];
  float *h_C = new float[height * width];

  deviceConfig();

  // generate data

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      h_A[i * width + j] = (i+j);
      h_B[i * width + j] = (i+j);
    }
  }

  cudaSetDevice(0);
  ERR_CHECK(cudaGetLastError());



  float *d_A, *d_B, *d_C;
  size_t pitch_A, pitch_B, pitch_C;

  cudaMallocPitch((void**)&d_A, &pitch_A,  (sizeof(float) * width), height);
  cudaMallocPitch((void**)&d_B, &pitch_B,  (sizeof(float) * width), height);
  cudaMallocPitch((void**)&d_C, &pitch_C,  (sizeof(float) * width), height);

  cudaMemcpy2DAsync(d_A, pitch_A, h_A, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
  cudaMemcpy2DAsync(d_B, pitch_B, h_B, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

  //int threadsPerBlock = 256;
  //int numBlocks = width / threadsPerBlock;

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + 15) / 16, (height + 15) / 16);
  //cudaDeviceSynchronize();
  addVector2D<<<numBlocks, threadsPerBlock>>>(d_A, pitch_A, d_B, pitch_B, d_C, pitch_C, width, height);


  cudaMemcpy2D(h_C, width * sizeof(float), d_C, pitch_C, width * sizeof(float), height, cudaMemcpyDeviceToHost);

   for(int i =0 ; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      std::cout << "Index " << i << ", " << j << ", Value " << h_C[i * width + j] << std::endl;
    }
  }

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

}


int main() {
  //oneDOperations();
  twoDOperations();
  return 0;
}