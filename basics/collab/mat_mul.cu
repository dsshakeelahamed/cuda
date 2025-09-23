%%writefile matrix_mul_1_0.cu
#include <iostream>
#include <cuda_runtime.h>

#define ERR_CHECK(err)                                                                                                  \
    if (err != cudaSuccess)                                                                                             \
    {                                                                                                                   \
        std::cout << "Error with cuda operation : " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        return;                                                                                                         \
    }

        __global__ void
        multiplyVector2D(float *A, size_t pitch_A, float *B, size_t pitch_B, float *C, size_t pitch_C, int width_A, int height_A, int width_B, int height_B, int width_C, int height_C)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    float *r_A = (float *)((char *)A + row * pitch_A);
    float *r_C = (float *)((char *)C + row * pitch_C);

    if (col < width_C && row < height_C)
    {
        float temp = 0.0f;
        for (int i = 0; i < width_A; i++)
        {
            float *r_B = (float *)((char *)B + i * pitch_B);
            temp += r_A[i] * r_B[col];
        }
        r_C[col] = temp;
    }

    return;
}
void deviceConfig()
{
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

void twoDOperations()
{
    size_t width_A = 1 << 12;
    size_t height_A = 1 << 12;
    size_t width_B = 1 << 12;
    size_t height_B = 1 << 12;
    size_t width_C = height_A;
    size_t height_C = width_B;
    // int N = width * height;
    // size_t size = N * sizeof(float);

    float *h_A = new float[height_A * width_A];
    float *h_B = new float[height_B * width_B];
    float *h_C = new float[height_C * width_C];

    deviceConfig();

    // generate data

    for (int i = 0; i < height_A; i++)
    {
        for (int j = 0; j < width_A; j++)
        {
            h_A[i * width_A + j] = (1);
        }
    }

    for (int i = 0; i < height_B; i++)
    {
        for (int j = 0; j < width_B; j++)
        {
            h_B[i * width_B + j] = (j);
        }
    }

    cudaSetDevice(0);
    ERR_CHECK(cudaGetLastError());

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStream_t stream_2;
    cudaStreamCreate(&stream_2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_A, *d_B, *d_C;
    size_t pitch_A, pitch_B, pitch_C;

    cudaMallocPitch((void **)&d_A, &pitch_A, (sizeof(float) * width_A), height_A);
    cudaMallocPitch((void **)&d_B, &pitch_B, (sizeof(float) * width_B), height_B);
    cudaMallocPitch((void **)&d_C, &pitch_C, (sizeof(float) * width_C), height_C);

    cudaEventRecord(start, 0);
    cudaMemcpy2DAsync(d_A, pitch_A, h_A, width_A * sizeof(float), width_A * sizeof(float), height_A, cudaMemcpyHostToDevice, stream);
    cudaMemcpy2DAsync(d_B, pitch_B, h_B, width_B * sizeof(float), width_B * sizeof(float), height_B, cudaMemcpyHostToDevice, stream_2);

    // int threadsPerBlock = 256;
    // int numBlocks = width / threadsPerBlock;

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width_C + 15) / 16, (height_C + 15) / 16);
    // cudaDeviceSynchronize();

    multiplyVector2D<<<numBlocks, threadsPerBlock, 0, stream>>>(d_A, pitch_A, d_B, pitch_B, d_C, pitch_C, width_A, height_A, width_B, height_B, width_C, height_C);
    cudaEventRecord(stop, 0);
    ERR_CHECK(cudaGetLastError());

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel took " << ms << " ms\n"
              << std::endl;

    cudaMemcpy2D(h_C, width_C * sizeof(float), d_C, pitch_C, width_C * sizeof(float), height_C, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
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

int main()
{
    twoDOperations();
    return 0;
}