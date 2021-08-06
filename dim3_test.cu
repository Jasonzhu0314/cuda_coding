#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void checkIndex(void)
{
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d) gridDim(%d,%d,%d)\n",
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char* argv[]) {
    
    int data_size = 32; 
    int thread_per_block_x = 32;
    if (argc > 2) {
        data_size = atoi(argv[1]);
        thread_per_block_x = atoi(argv[2]);
    }

    dim3 block(thread_per_block_x, 1);
    dim3 grid((data_size - 1) / thread_per_block_x + 1, 1);

    printf("block.x: %d, block.y: %d, block.z: %d\n", block.x, block.y, block.z);
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);
    checkIndex<<<grid,block>>>();

    cudaDeviceReset();
    return 0;
}