%%writefile blur.cu
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    }

__global__
void blurKernel(const unsigned char* input, unsigned char* output,
                int width, int height, int filterRadius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int filterSize = 2 * filterRadius + 1;
    int sum = 0, count = 0;

    for (int dy = -filterRadius; dy <= filterRadius; ++dy) {
        for (int dx = -filterRadius; dx <= filterRadius; ++dx) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            sum += input[ny * width + nx];
            count++;
        }
    }
    //output[y * width + x] = 128;
    output[y * width + x] = static_cast<unsigned char>(sum / count);
}

__global__ void blurKernel_2(const unsigned char* input, unsigned char* output,
                int width, int height, int filterRadius)
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int filterSize = 2 * filterRadius + 1;
    int sum = 0, count = 0;
    for (int dy = -filterRadius; dy <= filterRadius; ++dy) {
        for (int dx = -filterRadius; dx <= filterRadius; ++dx) {
          int nx = x + dx;
          int ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            continue;
          }
          sum += input[ny * width + nx];
          count++;
        }
      }
    output[y * width + x] = static_cast<unsigned char>(sum / count);
    }
  

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: ./blur input.bin output.bin width height [filterRadius]\n";
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    int width = std::stoi(argv[3]);
    int height = std::stoi(argv[4]);
    int filterRadius = (argc > 5) ? std::stoi(argv[5]) : 1;

    size_t imgSize = width * height;
    std::vector<unsigned char> h_input(imgSize), h_output(imgSize);

    std::ifstream in(inputFile, std::ios::binary);
    in.read(reinterpret_cast<char*>(h_input.data()), imgSize);
    in.close();

    unsigned char *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, imgSize));
    CHECK_CUDA(cudaMalloc(&d_output, imgSize));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), imgSize, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    std::cout << "Grid: " << grid.x << "x" << grid.y 
          << " Block: " << block.x << "x" << block.y << std::endl;
        
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    blurKernel<<<grid, block>>>(d_input, d_output, width, height, filterRadius);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout<<"Kernel took "<< ms << " ms\n" << std::endl ;
    //CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, imgSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++) {
        std::cout << (int)h_output[i] << " ";
    }
    std::cout << std::endl;

    std::ofstream out(outputFile, std::ios::binary);
    out.write(reinterpret_cast<char*>(h_output.data()), imgSize);
    out.close();

    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Blur done. Saved to " << outputFile << std::endl;
    return 0;
}
