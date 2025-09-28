%%writefile cuda_cnn.cu
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cstdlib>


// write a kernel for activation function
// write a kernel for dense layer

void ERR_CHECK(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl;
    exit(1);
  }
}


__global__ void relu_activation(float* input_matrix, int input_width, int input_height) {

  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < input_width && ty < input_height) {
    int index = ty * input_width + tx;
    input_matrix[index] = fmax(0.0f, input_matrix[index]);
  }
  return;
}

__global__ void dense_layer(float* matrixA, float* matrixB, float* matrixC, float* bias, int widthA, int heightA, int widthB, int heightB, int widthBias, int heigthBias)
{


//   assumption is that tileW and tileH are same
  extern __shared__ float sm[];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tileW = blockDim.x;
  int tileH = blockDim.y;

  // shared memory for matrix a
  float* smA = sm;
  // shared memory for matrix B
  float* smB = sm + (tileW * tileH);



  // Global output coordinates
  int row = blockIdx.y * tileW + ty;
  int col = blockIdx.x * tileH + tx;

  if (row >= heightA || col >= widthB) return;



  float sum = 0.0f;

  for (int k = 0; k < ((widthA + tileW - 1)/tileW); k++) {
    int linearIdx = ty * blockDim.x + tx;
    if (linearIdx < (tileW * tileH)) {
      int tileRow = linearIdx / tileW;
      int tileCol = linearIdx % tileW;

      int globalRowA = blockIdx.y * tileH + tileRow;
      int globalColA = k * tileW + tileCol;

      if (globalRowA < heightA && globalColA < widthA) {
        smA[tileRow * tileW + tileCol] = matrixA[globalRowA * widthA + globalColA];
      }
      else {
        smA[tileRow * tileW + tileCol] = 0.0f;
      }

      int globalRowB = k * tileW + tileRow;
      int globalColB = blockIdx.x * tileH + tileCol;

      if (globalRowB < heightB && globalColB < widthB) {
        smB[tileRow * tileW + tileCol] = matrixB[globalRowB * widthB + globalColB];
      }
      else {
        smB[tileRow * tileW + tileCol] = 0.0f;

    }
    }

  __syncthreads();

  for (int i = 0; i < tileW; i++) {
    sum += smA[ty * tileW + i] * smB[i * tileW + tx];
  }

  __syncthreads();
 }

if (row >= heightA || col >= widthB) {
    return;
  }

matrixC[row * widthB + col] = sum;


if (widthBias == 1 && heigthBias == heightA) {
  matrixC[row * widthB + col] += bias[col];
}

}

// __global__ void dense_layer_rectangle_sm(float* matrixA, float* matrixB, float* matrixC, float* bias, int widthA, int heightA, int widthB, int heightB, int widthBias, int heigthBias, int sm_M, int sm_K, int sm_N)
// {

//   extern __shared__ float sm[];
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;

//   int blockWidth = blockDim.x;
//   int blockHeight = blockDim.y;
//   int blockSize = blockWidth * blockHeight;


//   float* smA = sm;
//   float* smB = sm + (sm_M * sm_K);
//   int row = blockIdx.y * sm_M + ty;
//   int col = blockIdx.x * sm_N + tx;

//   //if (row >= heightA || col >= widthB) return;


//   //Basically need to loop over the elements and write accordingly

//   // use sm_M == sm_N
//   //int elementsToCompute = (sm_M + blockHeight - 1) / blockHeight;
//   int elementsToCompute = (sm_M * sm_K + blockSize - 1) / blockSize;                       
//   float sum[8];
//   for (int i = 0; i < 8; i++) {
//     sum[i] = 0.0f;
//   }


//   for (int k = 0; k < ((widthA + sm_K - 1)/sm_K); k++) {
//     for (int p=0; p < (sm_M * sm_K + blockSize - 1) / blockSize; p++) {
//       // get the position of element to load as a linear index
//       int linearIdx = (ty * blockWidth + tx) + p * blockSize ;
//       if (linearIdx < (sm_M * sm_K)) {
//         int tileRow = linearIdx / sm_K;
//         int tileCol = linearIdx % sm_K;

//         int globalRow = blockIdx.y * sm_M + tileRow;
//         int globalCol = k * sm_K + tileCol;

//         if (globalRow < heightA && globalCol < widthA) {
//         smA[tileRow * sm_K + tileCol] = matrixA[globalRow * widthA + globalCol];
//       }
//       else {
//         smA[tileRow * sm_K + tileCol] = 0.0f;
//       }
//     }
//     }

//     for (int p=0; p < (sm_K * sm_N + blockSize - 1) / blockSize; p++) {
//       int linearIdx = (ty * blockWidth + tx) + p * blockSize ;
//       if (linearIdx < (sm_K * sm_N)) {
//         int tileRow = linearIdx / sm_N;
//         int tileCol = linearIdx % sm_N;

//         int globalRow = k * sm_K + tileRow;
//         int globalCol = blockIdx.x * sm_N + tileCol;
//         if (globalRow < heightB && globalCol < widthB) {
//         smB[tileRow * sm_N + tileCol] = matrixB[globalRow * widthB + globalCol];
//     }
//       else {
//         smB[tileRow * sm_N + tileCol] = 0.0f;
//       }
//     }
//     }
//     __syncthreads();

//     // The bug is here, we are loading multiple elements per thread in sm
//     // However, we are only computing result of a single element
//     // so similar to p index, we should do the accumulation as well
//     // Major assumption is SM and SN are equal
//     //for (int i = 0; i < sm_K; i++) {
//     //  sum += smA[ty * sm_K + i] * smB[i * sm_N + tx];
//     // }

//     for (int i = 0; i < elementsToCompute; i++) {
//       int localRow = ty + i * blockHeight ;
//       if (localRow < sm_M) {
//         for (int j = 0; j < sm_K; j++) {
//           if (j < widthB) {
//             sum[i] += smA[localRow * sm_K + j] * smB[j * sm_N + tx ];
//           }
//         }
//       }
//     }

//     __syncthreads();
//     }

//   //matrixC[row * widthB + col] = sum;


//   for (int i = 0; i < elementsToCompute; i++) {
//     int globalRow = blockIdx.y * sm_M + ty + i * blockHeight;
//     int globalCol = blockIdx.x * sm_K + tx;
//     //if (globalRow == 0) printf("Global row = %d, col = %d\n",globalRow, globalCol);
//     if (globalRow < heightA && globalCol < widthB) {
      
//       matrixC[globalRow * widthB + globalCol] = sum[i] + bias[globalCol];
//     //if (globalRow == 0) printf("Global row = %d, col = %d, result = %f\n", globalRow, globalCol, matrixC[globalRow * widthB + globalCol]);
//     }
//   }


// //if (widthBias == 1 && heigthBias == heightA) {
// //  matrixC[row * widthB + col] += bias[col];
// //}

// }

void verify_dense_layer() {
  int widthA = 64;
  int heightA = 100;
  int widthB = 10;
  int heightB = 64;
  int widthBias = 1;
  int heigthBias = 100;

  float* matrixA = new float[widthA * heightA];
  float* matrixB = new float[widthB * heightB];
  float* matrixC = new float[widthA * heightB];
  float* matrixC_1 = new float[widthA * heightB];
  float* bias = new float[heightA];

  for (int i = 0; i < widthA * heightA; i++) {
    matrixA[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

  for (int i = 0; i < widthB * heightB; i++) {
    matrixB[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

  for (int i = 0; i < heigthBias; i++) {
    bias[i] = 1;
  }

  float *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixC_1, *d_bias;

  ERR_CHECK(cudaMalloc((void**)&d_matrixA, widthA * heightA * sizeof(float)));
  cudaMalloc((void**)&d_matrixB, widthB * heightB * sizeof(float));
  cudaMalloc((void**)&d_matrixC, heightA * widthB * sizeof(float));
  cudaMalloc((void**)&d_matrixC_1, heightA * widthB * sizeof(float));
  cudaMalloc((void**)&d_bias, heigthBias * widthBias * sizeof(float));

  ERR_CHECK(cudaMemcpy(d_matrixA, matrixA, widthA * heightA * sizeof(float), cudaMemcpyHostToDevice));
  cudaMemcpy(d_matrixB, matrixB, widthB * heightB * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, widthBias * heigthBias * sizeof(float), cudaMemcpyHostToDevice);

  int tileW = 16;
  int tileH = 16;
  dim3 blockSize (tileW, tileH);
  dim3 gridSize ((widthA + blockSize.x - 1) / blockSize.x, (heightB + blockSize.y - 1) / blockSize.y);

  size_t sharedMemSize = tileW * tileH * sizeof(float) * 2;
  printf("Shared memory size: %zu\n", sharedMemSize);
  printf("Calling kernel 1\n");
  dense_layer<<<gridSize, blockSize, sharedMemSize>>>(d_matrixA, d_matrixB, d_matrixC, d_bias, widthA, heightA, widthB, heightB, widthBias, heigthBias);

  ERR_CHECK(cudaMemcpy(matrixC, d_matrixC, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      std::cout << matrixC[i* widthB + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;


//   int sm_M = 64;
//   int sm_K = 16;
//   int sm_N = 64;

//   sharedMemSize = ((sm_M * sm_K) + ( sm_K * sm_N)) * sizeof(float);
//   printf("Shared memory size: %zu\n", sharedMemSize);
//   printf("Calling kernel 2\n");

//   dense_layer_rectangle_sm<<<gridSize, blockSize, sharedMemSize>>>(d_matrixA, d_matrixB, d_matrixC_1, d_bias, widthA, heightA, widthB, heightB, widthBias, heigthBias, sm_M, sm_K, sm_N);

//   ERR_CHECK(cudaGetLastError());
//   ERR_CHECK(cudaDeviceSynchronize());

//   ERR_CHECK(cudaMemcpy(matrixC_1, d_matrixC_1, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost));

//   for (int i = 0; i < 10; i++) {
//     for (int j = 0; j < 10; j++) {
//       std::cout << matrixC_1[i* widthB + j] << " ";
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;

  delete[] matrixA;
  delete[] matrixB;
  delete[] matrixC;
  delete[] matrixC_1;
  delete[] bias;
  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_matrixC);
  cudaFree(d_matrixC_1);
  cudaFree(d_bias);
}

void run_inference(std::string inputFile, size_t inputSize, std::string weightsFile_1, std::string biasFile_1, std::string weightsFile_2, std::string biasFile_2) {

  // assuming weight 1 is 784 x 128, weight 2 is 128 x 10, bias 1 is 128, bias 2 is 10
  size_t N = inputSize / 784;
  size_t weight1Size = 784 * 128;
  size_t weight2Size = 128 * 10;
  size_t bias1Size = 128;
  size_t bias2Size = 10;
  size_t outputElems = 10 * N;

  std::vector<float> input_data(inputSize);
  std::vector<float> weight1(weight1Size);
  std::vector<float> weight2(weight2Size);
  std::vector<float> bias1(bias1Size);
  std::vector<float> bias2(bias2Size);
  std::vector<float> output(outputElems);



  std::ifstream input_file(inputFile, std::ios::binary);
  std::ifstream weights_file_1(weightsFile_1, std::ios::binary);
  std::ifstream bias_file_1(biasFile_1, std::ios::binary);
  std::ifstream weights_file_2(weightsFile_2, std::ios::binary);
  std::ifstream bias_file_2(biasFile_2, std::ios::binary);

  input_file.read(reinterpret_cast<char*>(input_data.data()), inputSize * sizeof(float));
  weights_file_1.read(reinterpret_cast<char*>(weight1.data()), weight1Size * sizeof(float));
  bias_file_1.read(reinterpret_cast<char*>(bias1.data()), bias1Size * sizeof(float));
  weights_file_2.read(reinterpret_cast<char*>(weight2.data()), weight2Size * sizeof(float));
  bias_file_2.read(reinterpret_cast<char*>(bias2.data()), bias2Size * sizeof(float));

  // print a small sample to check if data is read correctly
  std::cout << "Input data: " << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << input_data[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Weight 1: " << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << weight1[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Bias 1: " << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << bias1[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Weight 2: " << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << weight2[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Bias 2: " << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << bias2[i] << " ";
  }
  std::cout << std::endl;

  input_file.close();
  weights_file_1.close();
  bias_file_1.close();
  weights_file_2.close();
  bias_file_2.close();


  float* d_input_data;
  float* d_weight1;
  float* d_weight2;
  float* d_bias1;
  float* d_bias2;
  float* d_layer1_output;
  float* d_layer2_output;

  cudaMalloc((void**)&d_input_data, inputSize * sizeof(float));
  cudaMalloc((void**)&d_weight1, weight1Size * sizeof(float));
  cudaMalloc((void**)&d_weight2, weight2Size * sizeof(float));
  cudaMalloc((void**)&d_bias1, bias1Size * sizeof(float));
  cudaMalloc((void**)&d_bias2, bias2Size * sizeof(float));
  cudaMalloc((void**)&d_layer1_output, (N * 128) * sizeof(float));
  cudaMalloc((void**)&d_layer2_output, (N * 10) * sizeof(float));

  cudaMemcpy(d_input_data, input_data.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight1, weight1.data(), weight1Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight2, weight2.data(), weight2Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias1, bias1.data(), bias1Size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias2, bias2.data(), bias2Size * sizeof(float), cudaMemcpyHostToDevice);

  int tileW = 16;
  int tileH = 16;
  dim3 block(tileW, tileH);
  dim3 grid1((128 + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  dim3 grid2((10 + block.x - 1) / block.x, (N + block.y - 1) / block.y);



  // call kernels in sequential manner for both the layers
  // validate the final output

  std::cout << "Calling kernel "<< std::endl;
  size_t sharedMemSize = tileW * tileH * sizeof(float) * 2;
  dense_layer<<<grid1, block, sharedMemSize>>>(d_input_data, d_weight1, d_layer1_output, d_bias1, 784, N, 128, 784, 1, 128);

//   int sm_M = 32;
//   int sm_K = 16;
//   int sm_N = 32;
//   grid1 = dim3((128 + sm_K - 1)/ sm_K, (N + sm_M - 1) / sm_M);
  //size_t sharedMemSize = ((sm_M * sm_K) + ( sm_K * sm_N)) * sizeof(float);
  //dense_layer_rectangle_sm<<<grid1, block, sharedMemSize>>>(d_input_data, d_weight1, d_layer1_output, d_bias1, 784, N, 128, 784, 1, 128, sm_M, sm_K, sm_N);
  ERR_CHECK(cudaGetLastError());
  ERR_CHECK(cudaDeviceSynchronize());

  std::vector<float> debug(N * 128);
  cudaMemcpy(debug.data(), d_layer1_output, N * 128 * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Layer 1 output: " << std::endl;
  for (int i = 0; i < 128; i++) {
    std::cout << debug[i] << " ";
  }
  std::cout << std::endl;

  relu_activation<<<grid1, block>>>(d_layer1_output, 128, N);

  ERR_CHECK(cudaGetLastError());
  ERR_CHECK(cudaDeviceSynchronize());

  cudaMemcpy(debug.data(), d_layer1_output, N * 128 * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Relu output: " << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << debug[i] << " ";
  }
  std::cout << std::endl;

  sharedMemSize = tileW * tileH * sizeof(float) * 2;
  dense_layer<<<grid2, block, sharedMemSize>>>(d_layer1_output, d_weight2, d_layer2_output, d_bias2, 128, N, 10, 128, 1, 10);
//   block = dim3(16, 16);
//   grid2 = dim3((10 + sm_K - 1)/ sm_K, (N + sm_M - 1) / sm_M);
  //dense_layer_rectangle_sm<<<grid2, block, sharedMemSize>>>(d_layer1_output, d_weight2, d_layer2_output, d_bias2, 128, N, 10, 128, 1, 10, sm_M, sm_K, sm_N);

  ERR_CHECK(cudaGetLastError());
  ERR_CHECK(cudaDeviceSynchronize());


  cudaMemcpy(output.data(), d_layer2_output, outputElems * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Output: " << std::endl;
  // local softmax
  float max_out {0.0f};
  int result {-1};
  for (int i = 0; i < 5; i++) {
    max_out = 0.0f;
    result = -1;
    for (int  j = 0; j < 10; j++) {
      std::cout << output[i*10+j] << " ";
      if (output[i*10 + j] > max_out) {
        max_out = output[i*10 + j];
        result = j;
    }
    }
    std::cout << std::endl;
    std::cout << "Result for " << i << " image is " << result << " with resulting confidence " << max_out << std::endl;
  }


  cudaFree(d_input_data);
  cudaFree(d_weight1);
  cudaFree(d_weight2);
  cudaFree(d_bias1);
  cudaFree(d_bias2);
  cudaFree(d_layer1_output);
  cudaFree(d_layer2_output);


}

int main(int argc, char** argv) {



  //verify_dense_layer();

  if (argc != 7) {
    std::cerr << "Usage: ./cuda_cnn inputFile inputSize weightsFile_1 biasFile_1 weightsFile_2 biasFile_2\n"<< std::endl;
  }
    cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  std::string inputFile = argv[1];
  size_t inputSize = std::stoi(argv[2]);
  std::string weightsFile_1 = argv[3];
  std::string biasFile_1 = argv[4];
  std::string weightsFile_2 = argv[5];
  std::string biasFile_2 = argv[6];
  std::cout << "Calling inference " << std::endl;


  run_inference(inputFile, inputSize, weightsFile_1, biasFile_1, weightsFile_2, biasFile_2);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  std::cout << "Inference took " << ms << " ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

}