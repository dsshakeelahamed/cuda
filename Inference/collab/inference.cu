%%writefile cuda_cnn.cu
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cublas_v2.h>
#include <cublasLt.h>


// write a kernel for activation function
// write a kernel for dense layer

void ERR_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl;
        exit(1);
    }
}


__global__ void relu_activation(float* input_matrix, int input_width, int input_height) {

    // TODO: needs to be tested, proceed only after testing these
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

    // TODO: needs to be verified
    // this is just mat_mul, you can also add bias in the same, simple addition to the respective element
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
    int row = blockIdx.y * tileH + ty;
    int col = blockIdx.x * tileW + tx;

    float sum = 0.0f;

    for (int k = 0; k < ((widthA + tileW - 1) / tileW); k++) {
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
            int globalColB = blockIdx.x * tileW + tileCol;

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

__global__ void dense_layer_rectangle_sm(float* matrixA, float* matrixB, float* matrixC, float* bias, int widthA, int heightA, int widthB, int heightB, int widthBias, int heigthBias, int sm_M, int sm_K, int sm_N)
{

    extern __shared__ float sm[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;
    int blockSize = blockWidth * blockHeight;


    float* smA = sm;
    float* smB = sm + (sm_M * sm_K);



    //Basically need to loop over the elements and write accordingly

    // use sm_M == sm_N
    //int elementsToCompute = (sm_M + blockHeight - 1) / blockHeight;
    int elementsToCompute = (sm_M * sm_N) / blockSize;
    int row_elements = sm_M / blockHeight;
    int col_elements = sm_N / blockWidth;
    float sum[8];
    for (int i = 0; i < 8; i++) {
        sum[i] = 0.0f;
    }


    for (int k = 0; k < ((widthA + sm_K - 1) / sm_K); k++) {

        // the problem here is linear indices being used for p
        // turn it into a nested for loop where you move elements to compute times, as simple as that
        // same loop can be used for computing B value also
        for (int p_r = 0; p_r < (sm_M / blockHeight); p_r++) {
            for (int p_c = 0; p_c < (sm_K / blockWidth); p_c++) {
                int linearIdx = (p_r * blockHeight + ty) * blockWidth + (p_c * blockWidth + tx);
                if (linearIdx < (sm_M * sm_K)) {
                    int tileRowA = linearIdx / sm_K;
                    int tileColA = linearIdx % sm_K;
                    // assumption that sm_K == blockWidth (need to check more on how to remove this)

                    int globalRowA = blockIdx.y * sm_M + tileRowA;
                    int globalColA = k * sm_K + tileColA;

                    if (globalRowA < heightA && globalColA < widthA) {
                        smA[tileRowA * sm_K + tileColA] = matrixA[globalRowA * widthA + globalColA];
                    }
                    else {
                        smA[tileRowA * sm_K + tileColA] = 0.0f;
                    }
                }
            }
        }
        int count = 0;

        for (int p_r = 0; p_r < (sm_N / blockWidth); p_r++) {
            for (int p_c = 0; p_c < (sm_K / blockHeight); p_c++) {
                int linearIdx = (p_r * blockHeight + ty) * blockWidth + (p_c * blockWidth + tx);
                if (linearIdx < (sm_K * sm_N)) {

                    int tileRowB = linearIdx / sm_N;
                    int tileColB = linearIdx % sm_N;


                    int globalRowB = k * sm_K + tileRowB;
                    int globalColB = blockIdx.x * sm_N + tileColB;

                    if (globalRowB < heightB && globalColB < widthB) {

                        smB[tileRowB * sm_N + tileColB] = matrixB[globalRowB * widthB + globalColB];
                    }
                    else {
                        //printf("incorrect values tR=%d, tC=%d, gR=%d, gC=%d\n", tileRowB, tileColB, globalRowB, globalColB);
                        smB[tileRowB * sm_N + tileColB] = 0.0f;
                    }
                }
                //else {
                //    printf("linearIdx = %d, tx=%d, ty=%d, p_r=%d, p_c=%d\n", linearIdx, tx,ty,p_r,p_c);
                //}
            }
        }


        __syncthreads();

        // The bug is here, we are loading multiple elements per thread in sm
        // However, we are only computing result of a single element
        // so similar to p index, we should do the accumulation as well
        // Major assumption is SM and SN are equal

        for (int i = 0; i < (sm_M / blockHeight); i++) {
            for (int j = 0; j < (sm_N / blockWidth); j++) {
                int localRow = ty + i * blockHeight;
                int localCol = tx + j * blockWidth;
                if (localRow < sm_M && localCol < sm_N) {

                    for (int inner_k = 0; inner_k < sm_K; inner_k++) {
                        sum[i * col_elements + j] += smA[localRow * sm_K + inner_k] * smB[inner_k * sm_N + localCol];

                    }
                }
            }
        }

        __syncthreads();

    }


    for (int i = 0; i < (sm_M / blockHeight); i++) {
        for (int j = 0; j < (sm_N / blockWidth); j++) {
            int globalRow = blockIdx.y * sm_M + ty + i * blockHeight;
            int globalCol = blockIdx.x * sm_N + tx + j * blockWidth;
            if (globalRow < heightA && globalCol < widthB) {
                matrixC[globalRow * widthB + globalCol] = sum[i * col_elements + j] + bias[globalCol];
            }
        }
    }
    if (tx == 0 && ty == 0) {
        //for (int i=0; i< sm_M; i++) {
        //    for (int j =0; j < sm_K; j++) {
        //        printf("sm_A value at %d, %d = %f\n", i, j, smA[i*sm_K + j]);
        //    }
        //}
        //for (int i=0; i< sm_K; i++) {
        //    for (int j =0; j < sm_N; j++) {
        //        printf("sm_B value at %d, %d = %f\n", i, j, smB[i*sm_N + j]);
        //    }
        //}
    }
}

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
    float* matrixC_2 = new float[widthA * heightB];
    float* bias = new float[heightA];

    for (int i = 0; i < widthA * heightA; i++) {
        matrixA[i] = static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
        //matrixA[i] = 1;
    }

    for (int i = 0; i < widthB * heightB; i++) {
        matrixB[i] = static_cast <float>(rand()) / static_cast <float>(RAND_MAX);
        //matrixB[i] = 1;
    }

    for (int i = 0; i < heigthBias; i++) {
        bias[i] = 0;
    }

    float* d_matrixA, * d_matrixB, * d_matrixC, * d_matrixC_1, * d_matrixC_2, * d_bias;

    ERR_CHECK(cudaMalloc((void**)&d_matrixA, widthA * heightA * sizeof(float)));
    cudaMalloc((void**)&d_matrixB, widthB * heightB * sizeof(float));
    cudaMalloc((void**)&d_matrixC, heightA * widthB * sizeof(float));
    cudaMalloc((void**)&d_matrixC_1, heightA * widthB * sizeof(float));
    cudaMalloc((void**)&d_matrixC_2, heightA * widthB * sizeof(float));
    cudaMalloc((void**)&d_bias, heigthBias * widthBias * sizeof(float));

    ERR_CHECK(cudaMemcpy(d_matrixA, matrixA, widthA * heightA * sizeof(float), cudaMemcpyHostToDevice));
    cudaMemcpy(d_matrixB, matrixB, widthB * heightB * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, widthBias * heigthBias * sizeof(float), cudaMemcpyHostToDevice);

    int tileW = 16;
    int tileH = 16;
    dim3 blockSize(tileW, tileH);
    dim3 gridSize((widthB + blockSize.x - 1) / blockSize.x, (heightA + blockSize.y - 1) / blockSize.y);

    size_t sharedMemSize = tileW * tileH * sizeof(float) * 2;
    printf("Shared memory size: %zu\n", sharedMemSize);
    printf("Calling kernel 1\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dense_layer << <gridSize, blockSize, sharedMemSize >> > (d_matrixA, d_matrixB, d_matrixC, d_bias, widthA, heightA, widthB, heightB, widthBias, heigthBias);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    ERR_CHECK(cudaGetLastError());
    ERR_CHECK(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Dense layer took " << ms << " ms" << std::endl;
    ERR_CHECK(cudaMemcpy(matrixC, d_matrixC, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << matrixC[i * widthB + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;


    cudaEvent_t start_r, stop_r;
    cudaEventCreate(&start_r);
    cudaEventCreate(&stop_r);
    int sm_M = 32;
    int sm_K = 16;
    int sm_N = 32;

    sharedMemSize = ((sm_M * sm_K) + (sm_K * sm_N)) * sizeof(float);
    printf("Shared memory size: %zu\n", sharedMemSize);
    printf("Calling kernel 2\n");
    gridSize = dim3((widthB + sm_N - 1) / sm_N, (heightA + sm_M - 1) / sm_M);
    cudaEventRecord(start_r);
    dense_layer_rectangle_sm << <gridSize, blockSize, sharedMemSize >> > (d_matrixA, d_matrixB, d_matrixC_2, d_bias, widthA, heightA, widthB, heightB, widthBias, heigthBias, sm_M, sm_K, sm_N);
    cudaEventRecord(stop_r);
    cudaEventSynchronize(stop_r);
    ERR_CHECK(cudaGetLastError());
    ERR_CHECK(cudaDeviceSynchronize());
    ms = 0.0f;
    cudaEventElapsedTime(&ms, start_r, stop_r);
    std::cout << "Rectangular matrix layer took " << ms << " ms" << std::endl;
    ERR_CHECK(cudaMemcpy(matrixC_2, d_matrixC_2, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << matrixC_2[i * widthB + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    printf("Calling Cublas \n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start_c, stop_c;
    cudaEventCreate(&start_c);
    cudaEventCreate(&stop_c);

    cudaEventRecord(start_c);
    cublasStatus_t stat = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        widthB,       // m (columns of C)
        heightA,       // n (rows of C)
        widthA,       // k
        &alpha,
        d_matrixB, widthB,   // B (K x N)
        d_matrixA, widthA,    // A (M x K)
        &beta,
        d_matrixC_1, widthB    // C (M x N)
    );
    cudaEventRecord(stop_c);
    cudaEventSynchronize(stop_c);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS SGEMM failed!\n";
    }

    ms = 0.0f;
    cudaEventElapsedTime(&ms, start_c, stop_c);
    std::cout << "Cublas took " << ms << " ms" << std::endl;
    ERR_CHECK(cudaMemcpy(matrixC_1, d_matrixC_1, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << matrixC_1[i * widthB + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start_r);
    cudaEventDestroy(stop_r);
    cudaEventDestroy(start_c);
    cudaEventDestroy(stop_c);
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

    std::cout << "Calling kernel " << std::endl;

    cudaEvent_t layer1Start, layer1Stop;
    cudaEventCreate(&layer1Start);
    cudaEventCreate(&layer1Stop);

    cudaEventRecord(layer1Start);
    //size_t sharedMemSize = tileW * tileH * sizeof(float) * 2;
    //dense_layer<<<grid1, block, sharedMemSize>>>(d_input_data, d_weight1, d_layer1_output, d_bias1, 784, N, 128, 784, 1, 128);

    int sm_M = 64;
    int sm_K = 16;
    int sm_N = 64;
    grid1 = dim3((128 + sm_N - 1) / sm_N, (N + sm_M - 1) / sm_M);
    size_t sharedMemSize = ((sm_M * sm_K) + (sm_K * sm_N)) * sizeof(float);

    dense_layer_rectangle_sm << <grid1, block, sharedMemSize >> > (d_input_data, d_weight1, d_layer1_output, d_bias1, 784, N, 128, 784, 1, 128, sm_M, sm_K, sm_N);

    cudaEventRecord(layer1Stop);
    ERR_CHECK(cudaGetLastError());
    ERR_CHECK(cudaDeviceSynchronize());

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, layer1Start, layer1Stop);
    std::cout << "Dense layer 1 took " << ms << " ms" << std::endl;
    std::vector<float> debug(N * 128);
    cudaMemcpy(debug.data(), d_layer1_output, N * 128 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Layer 1 output: " << std::endl;
    for (int i = 0; i < 128; i++) {
        std::cout << debug[i] << " ";
    }
    std::cout << std::endl;

    relu_activation << <grid1, block >> > (d_layer1_output, 128, N);

    ERR_CHECK(cudaGetLastError());
    ERR_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(debug.data(), d_layer1_output, N * 128 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Relu output: " << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << debug[i] << " ";
    }
    std::cout << std::endl;

    cudaEvent_t layer2Start, layer2Stop;
    cudaEventCreate(&layer2Start);
    cudaEventCreate(&layer2Stop);

    //sharedMemSize = tileW * tileH * sizeof(float) * 2;
    //dense_layer<<<grid2, block, sharedMemSize>>>(d_layer1_output, d_weight2, d_layer2_output, d_bias2, 128, N, 10, 128, 1, 10);
    sm_M = 32;
    sm_N = 32;
    block = dim3(16, 16);
    grid2 = dim3((10 + sm_N - 1) / sm_N, (N + sm_M - 1) / sm_M);
    cudaEventRecord(layer2Start);
    dense_layer_rectangle_sm << <grid2, block, sharedMemSize >> > (d_layer1_output, d_weight2, d_layer2_output, d_bias2, 128, N, 10, 128, 1, 10, sm_M, sm_K, sm_N);
    cudaEventRecord(layer2Stop);
    ERR_CHECK(cudaGetLastError());
    ERR_CHECK(cudaDeviceSynchronize());

    //relu_activation<<<grid2, block>>>(d_layer2_output, 10, N);
    ms = 0.0f;
    cudaEventElapsedTime(&ms, layer2Start, layer2Stop);
    std::cout << "Dense layer 2 took " << ms << " ms" << std::endl;


    cudaMemcpy(output.data(), d_layer2_output, outputElems * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output: " << std::endl;
    // local softmax
    float max_out{ 0.0f };
    int result{ -1 };
    for (int i = 0; i < 10; i++) {
        max_out = 0.0f;
        result = -1;
        for (int j = 0; j < 10; j++) {
            std::cout << output[i * 10 + j] << " ";
            if (output[i * 10 + j] > max_out) {
                max_out = output[i * 10 + j];
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
    cudaEventDestroy(layer1Start);
    cudaEventDestroy(layer1Stop);
    cudaEventDestroy(layer2Start);
    cudaEventDestroy(layer2Stop);

}

int main(int argc, char** argv) {



    //verify_dense_layer();

    if (argc != 7) {
        std::cerr << "Usage: ./cuda_cnn inputFile inputSize weightsFile_1 biasFile_1 weightsFile_2 biasFile_2\n" << std::endl;
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