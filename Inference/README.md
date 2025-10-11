The kernels have been executed on google collab T4 gpu, can be done with following commands

### Add kernel
`!nvcc -arch=sm_75 cuda_cnn.cu -o cuda_cnn` \
`!./cuda_cnn test_inputs.bin <784*test_sample_size> weights1.bin bias1.bin weights2.bin bias2.bin`

### Metric Comparison by Layer 
| Iteration | Layer | Kernel | M | K | N | FLOPs (2 x M x N x K) | Time (ms) | GFLOPS/s |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:--:|
| 1 | Dense 1 | SM Kernel | 17500 | 784 | 128 | 3.51 × 10⁹ | 13.1455 | 267.017|
| 2 | Dense 1 | SM Kernel | 17500 | 784 | 128 | 3.51 × 10⁹ | 13.1027 | 267.88 |
| 3 | Dense 1 | SM Kernel | 17500 | 784 | 128 | 3.51 × 10⁹ | 13.1172 | 267.58 |
| 1 | Dense 1 | SM Kernel (Block Tiling) | 17500 | 784 | 128 | 3.51 × 10⁹ | 11.096 | 316.33 |
| 2 | Dense 1 | SM Kernel (Block Tiling) | 17500 | 784 | 128 | 3.51 × 10⁹ | 11.087 | 316.58 |
| 3 | Dense 1 | SM Kernel (Block Tiling) | 17500 | 784 | 128 | 3.51 × 10⁹ | 11.103 | 316.13 |
|  |  |  |  |  |  |  |  |  |
| 1 | Dense 2 | SM Kernel | 17500 | 128 | 10 | 4.48 × 10⁷ | 0.289 | 155.017 |
| 2 | Dense 2 | SM Kernel | 17500 | 128 | 10 | 4.48 × 10⁷ | 0.3065 | 146.166 |
| 3 | Dense 2 | SM Kernel | 17500 | 128 | 10 | 4.48 × 10⁷ | 0.2947 | 152.019 |
| 1 | Dense 2 | SM Kernel (Block Tiling) | 17500 | 128 | 10 | 4.48 × 10⁷ | 0.5536 | 80.925 |
| 2 | Dense 2 | SM Kernel (Block Tiling) | 17500 | 128 | 10 | 4.48 × 10⁷ | 0.5526 | 81.071 |
| 3 | Dense 2 | SM Kernel (Block Tiling) | 17500 | 128 | 10 | 4.48 × 10⁷ | 0.5529 | 81.027 |