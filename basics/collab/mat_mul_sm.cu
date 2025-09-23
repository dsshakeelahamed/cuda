%%writefile matrix_mul_shared_mem_1_0.cu
#include <iostream>
#include <cuda_runtime.h>

#define ERR_CHECK(err) \
  if (err != cudaSuccess) { \
    std::cout << "Error with cuda operation : " << cudaGetErrorString(err) << " at line " << __LINE__ <<std::endl; \
    return; \
  } \

__global__ void multiplyVector2DShared(
    float* A, size_t pitch_A,
    float* B, size_t pitch_B,
    float* C, size_t pitch_C,
    int width_A, int height_A,
    int width_B, int height_B,
    int width_C, int height_C,
    int tile_sizes)
{
    // Fixed tile size
    const int tile_size = 16;

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column indices
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    // Static shared memory tiles
    __shared__ float tile_A[tile_size][tile_size];
    __shared__ float tile_B[tile_size][tile_size];

    // Accumulator in registers
    float temp = 0.0f;

    // Loop over tiles along K dimension
    for (int k = 0; k < width_A; k += tile_size)
    {
        // Load tile of A into shared memory
        if (row < height_A && (k + tx) < width_A)
            tile_A[ty][tx] = ((float*)((char*)A + row*pitch_A))[k + tx];
        else
            tile_A[ty][tx] = 0.0f;

        // Load tile of B into shared memory
        if ((k + ty) < height_B && col < width_B)
            tile_B[ty][tx] = ((float*)((char*)B + (k + ty)*pitch_B))[col];
        else
            tile_B[ty][tx] = 0.0f;

        __syncthreads(); // ensure tile is fully loaded

        // Compute partial sums for this tile
        for (int i = 0; i < tile_size; ++i)
        {
            temp += tile_A[ty][i] * tile_B[i][tx];
        }

        __syncthreads(); // wait before next tile load
    }

    // Write final result to C
    if (row < height_C && col < width_C)
        ((float*)((char*)C + row*pitch_C))[col] = temp;
}



__global__ void multiplyVector2DSharedOld(float* A, size_t pitch_A, float* B, size_t pitch_B, float* C, size_t pitch_C, int width_A, int height_A, int width_B, int height_B, int width_C, int height_C) {


  const int tile_size = 16;

  // define shared memory
  __shared__ float tile_A [tile_size][tile_size];
  __shared__ float tile_B [tile_size][tile_size];

  // get the indices for resultant matrix
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;


  // initialize a temporary variable to store partial elements
  float temp = 0.0f;

  // k is basically the dimension of the tiles you are working with
  // within a thread block, you've got each threads having id's within the range of tile size
  //
  for (int k=0; k < (width_A + tile_size -1)/tile_size; k++) {
    // initalizing tile values to be default 0
    tile_A[threadIdx.y][threadIdx.x] = 0.0f;
    tile_B[threadIdx.y][threadIdx.x] = 0.0f;


    // Here you are checking if the row you are working is within bounds for A, whereas column is calculated using the tile size
    if ( row < height_A && (k*tile_size+threadIdx.x) < width_A){
      float* r_A = (float*)((char*)A + row * pitch_A);
      tile_A[threadIdx.y][threadIdx.x] = r_A[tile_size * k + threadIdx.x];
    }
    if ((k*tile_size + threadIdx.y) < height_B && col < width_B){
      float* r_B = (float*)((char*)B + (tile_size * k + threadIdx.y) * pitch_B);
      tile_B[threadIdx.y][threadIdx.x] = r_B[col];
    }
      __syncthreads();

      for (int i=0 ; i < tile_size; i++) {
        temp += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
      }
      __syncthreads();
  }



  if (col < width_C && row < height_C) {
    float* r_C = (float*)((char*)C + row * pitch_C);
    r_C[col] = temp;
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


void twoDOperations() {
  size_t width_A = 1 << 12;
  size_t height_A = 1 << 12;
  size_t width_B = 1 << 12;
  size_t height_B = 1 << 12;
  size_t width_C = height_A;
  size_t height_C = width_B;
  //int N = width * height;
  //size_t size = N * sizeof(float);

  float *h_A = new float[height_A * width_A];
  float *h_B = new float[height_B * width_B];
  float *h_C = new float[height_C * width_C];

  deviceConfig();

  // generate data

  for (int i = 0; i < height_A; i++) {
    for (int j = 0; j < width_A; j++) {
      h_A[i * width_A + j] = (1);
    }
  }

  for (int i = 0; i < height_B; i++) {
    for (int j = 0; j < width_B; j++) {
      h_B[i * width_B + j] = (j);
    }
  }

  cudaSetDevice(0);
  ERR_CHECK(cudaGetLastError());

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  float *d_A, *d_B, *d_C;
  size_t pitch_A, pitch_B, pitch_C;

  cudaMallocPitch((void**)&d_A, &pitch_A,  (sizeof(float) * width_A), height_A);
  cudaMallocPitch((void**)&d_B, &pitch_B,  (sizeof(float) * width_B), height_B);
  cudaMallocPitch((void**)&d_C, &pitch_C,  (sizeof(float) * width_C), height_C);

  cudaEventRecord(start, stream);
  cudaMemcpy2DAsync(d_A, pitch_A, h_A, width_A * sizeof(float), width_A * sizeof(float), height_A, cudaMemcpyHostToDevice, stream);
  cudaMemcpy2DAsync(d_B, pitch_B, h_B, width_B * sizeof(float), width_B * sizeof(float), height_B, cudaMemcpyHostToDevice, stream);

  //int threadsPerBlock = 256;
  //int numBlocks = width / threadsPerBlock;


  //cudaDeviceSynchronize();

  int tile_size = 16;
  dim3 threadsPerBlock(tile_size, tile_size);
  dim3 numBlocks((width_C + tile_size -1) / tile_size, (height_C + tile_size-1) / tile_size);
  multiplyVector2DSharedOld<<<numBlocks, threadsPerBlock>>>(d_A, pitch_A, d_B, pitch_B, d_C, pitch_C, width_A, height_A, width_B, height_B, width_C, height_C);
  cudaEventRecord(stop, stream);
  ERR_CHECK(cudaGetLastError());

  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  std::cout<<"Kernel took "<< ms << " ms\n" << std::endl ;

  cudaMemcpy2D(h_C, width_C * sizeof(float), d_C, pitch_C, width_C * sizeof(float), height_C, cudaMemcpyDeviceToHost);

   for(int i =0 ; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      std::cout << "Index " << i << ", " << j << ", Value " << h_C[i * width_C + j] << std::endl;
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
  twoDOperations();
  return 0;
}