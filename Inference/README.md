The kernels have been executed on google collab T4 gpu, can be done with following commands

### Add kernel
`!nvcc -arch=sm_75 cuda_cnn.cu -o cuda_cnn` \
`!./cuda_cnn test_inputs.bin <784*test_sample_size> weights1.bin bias1.bin weights2.bin bias2.bin`