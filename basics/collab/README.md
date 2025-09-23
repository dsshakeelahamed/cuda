The kernels have been executed on google collab T4 gpu, can be done with following commands

### Add kernel
`!nvcc -arch=sm_75 vector_add_1_0.cu -o vector_add_1_0` \
`!./vector_add_1_0`