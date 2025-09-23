%%writefile color_blur.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>



void ERR_CHECK(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl;
    exit(1);
  }
}

__global__ void colorBlurKernel(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  int filterSize = 2 * radius + 1;

  int sum[3] {0,0,0};
  int count {0};
  for (int dy = -radius; dy < radius; dy++) {
    for (int dx = -radius; dx < radius; dx++) {
      int nx = min(max(x + dx, 0), width - 1);
      int ny = min(max(y + dy, 0), height - 1);
      for (int c = 0; c < channels; c++ ) {
        sum[c] += input[(ny * width + nx) * channels + c];
      }
      count++;
    }
  }
  for (int c =0 ; c< channels; c++) {
    output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum[c] / count);
  }
}

  
__global__ void colorBlurSharedMemory(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius) 
{
  __shared__ unsigned char pixelData [32][32][3];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  //if (x >= width || y >= height) return;
  if (x >= width) 
   x = width - 1;
  if (y >= height) 
   y = height - 1;


  for(int c=0; c<channels; c++) {
    pixelData[ty][tx][c] = input[(y * width + x) * channels + c];
  }

  __syncthreads();
  
  //if (tx < radius || tx > radius + 16 - 1 || ty < radius || ty > radius + 16 - 1) {
    // this is halo threads, because of this, there's a mismatch in final output
    // i believe, if this is uncommented, some importnat pixels on borders miss out
    // if commented, some pixels get overwritten with incorrect data, 
    // sit and layout how it should look
    //return;
    
  //}
  int blockOut = 16;

  bool isInteriorX = (tx >= radius) && (tx < (radius + blockOut));
  bool isInteriorY = (ty >= radius) && (ty < (radius + blockOut));
  if (! (isInteriorX && isInteriorY)) {
    return; // halo loader only, done
  }


  int sum[3] {0,0,0};
  int count {0};
  for (int dx = -radius; dx <= radius; dx++) {
    for (int dy = -radius ; dy <= radius; dy++) {
      int nx = min(max(tx + dx, 0), blockDim.x + 2*radius - 1);
      int ny = min(max(ty + dy, 0), blockDim.y + 2*radius - 1);
      for (int c = 0; c < channels; c++ ) {
        sum[c] += pixelData[ny][nx][c];
      }
      count++;
    }
  }

  //int outX = blockIdx.x * blockOut + (tx - radius); // compute output pixel x
  //int outY = blockIdx.y * blockOut + (ty - radius);
  //if (outX < width && outY < height) {
    //int outIdx = (outY * width + outX) * channels;
    for (int c =0 ; c< channels; c++) {
        //output[outIdx + c] = static_cast<unsigned char>(sum[c]/count);
        output[(y * width + x) * channels + c] = static_cast<unsigned char>(sum[c]/count);
    }
  //}
}

// static-shared example; adapt to extern __shared__ easily
#define MAX_CHANNELS 4

__global__ void colorBlurSharedMemOverProvision(unsigned char* input, unsigned char* output, int width, int height, int channels, int radius, int tileSize) {

  extern __shared__ unsigned char pixelData[];
  //unsigned char* pixelData = sm;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int haloWidth = blockDim.x;
  int haloHeight = blockDim.y;

  // global tile block you are working with
  int globalBlockX = blockIdx.x * tileSize;
  int globalBlockY = blockIdx.y * tileSize;

  // global pixel indexes, radius is subracted to get the halo as well
  int gx = globalBlockX - radius + tx;
  int gy = globalBlockY - radius + ty;

  // clamping

  int gx_clamped = min(max(gx, 0), width - 1 );
  int gy_clamped = min(max(gy, 0), height -1 );

  // load pixel data into shared memory
  int pixelDataIndex = (ty * haloWidth + tx) * channels;
  for (int c = 0; c < channels; c++) {
    pixelData[pixelDataIndex + c] = input[(gy_clamped * width + gx_clamped) * channels + c];
  }
  __syncthreads();
  
  // now check if interior or halo

  bool isInteriorX = (tx >= radius) && (tx < (radius + tileSize));
  bool isInteriorY = (ty >= radius) && (ty < (radius + tileSize));

  if (! (isInteriorX && isInteriorY)) {
    return; // halo loader only, done
  }

  int sum[3] {0,0,0};
  int count {0};

  for (int dx = -radius; dx <= radius; dx++) {
    for (int dy = -radius ; dy <= radius; dy++) {
      int nx = min(max(tx + dx, 0), haloWidth - 1);
      int ny = min(max(ty + dy, 0), haloHeight - 1);
      for (int c = 0; c < channels; c++ ) {
        sum[c] += pixelData[(ny * haloWidth + nx) * channels + c];
      }
      count++;
    }
  }
  for (int c =0 ; c< channels; c++) {
    output[(gy_clamped * width + gx_clamped) * channels + c] = static_cast<unsigned char>(sum[c]/count);
  }


}


int main(int argc, char** argv) {
// totally 7 args
// executable, input, output, width, height, channels, radius
if (argc < 7) {
  std::cerr << "Usage: ./color_blur input.bin output.bin width height channels radius\n";
  return 1;
}

std::string inputFile = argv[1];
std::string outputFile = argv[2];
int width = std::stoi(argv[3]);
int height = std::stoi(argv[4]);
int channels = std::stoi(argv[5]);
int radius = std::stoi(argv[6]);

size_t imgSize = width * height * channels;

size_t max_val = static_cast<size_t>(-1);
std::cout << "Maximum value of size_t: " << max_val << " " << imgSize << std::endl;

std::vector<unsigned char> h_input(imgSize), h_output(imgSize);

std::ifstream inputStream(inputFile, std::ios::binary);
inputStream.read(reinterpret_cast<char*>(h_input.data()), imgSize);
inputStream.close();

unsigned char *d_input, *d_output;
ERR_CHECK(cudaMalloc(&d_input, imgSize));
ERR_CHECK(cudaMalloc(&d_output, imgSize));

ERR_CHECK(cudaMemcpy(d_input, h_input.data(), imgSize, cudaMemcpyHostToDevice));
const int blockSize = 16;
//dim3 block(blockSize + 2*radius, blockSize + 2*radius);
//dim3 grid((width + blockSize - 1) / blockSize,
//         (height + blockSize - 1) / blockSize);  

//std::cout << "Grid: " << grid.x << "x" << grid.y
//          << " Block: " << block.x << "x" << block.y << std::endl;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);

//colorBlurKernel<<<grid, block>>>(d_input, d_output, width, height, channels, radius);
//colorBlurSharedMemory<<<grid, block>>>(d_input, d_output, width, height, channels, radius);

int blockOut = 16;
dim3 block(blockOut + 2*radius, blockOut + 2*radius);
dim3 grid( (width + blockOut - 1)/blockOut, (height + blockOut - 1)/blockOut );

// compute shared mem size in bytes
size_t sharedBytes = (block.x * block.y * channels) * sizeof(unsigned char);

// launch
//colorBlurOverProvision<<<grid, block, sharedBytes>>>(d_input, d_output, width, height, channels, radius, blockOut);
colorBlurSharedMemOverProvision<<<grid, block, sharedBytes>>>(d_input, d_output, width, height, channels, radius, blockOut);



cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
std::cout << "Kernel took " << ms << " ms\n" << std::endl;

ERR_CHECK(cudaMemcpy(h_output.data(), d_output, imgSize, cudaMemcpyDeviceToHost));

  for (int i = 0; i < 10; i++) {
      std::cout << (int)h_output[i] << " ";
  }
  std::cout << std::endl;

std::ofstream outputStream(outputFile, std::ios::binary);
outputStream.write(reinterpret_cast<char*>(h_output.data()), imgSize);
outputStream.close();

cudaFree(d_input);
cudaFree(d_output); 




}