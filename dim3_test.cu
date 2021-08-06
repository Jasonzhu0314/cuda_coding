#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "time_utils.h"


int main(int argc, char* argv[]) {
    int block_size = atoi(argv[1]);

    double istart = CpuSecond();
    int data_size = 32; 
    dim3 block(block_size, 1);
    dim3 grid((data_size - 1) / block_size + 1, 1);

    printf("block.x: %d, block.y: %d\n", block.x, block.y);
    printf("grid.x: %d, grid.y: %d\n", grid.x, grid.y);

    double ielpase = CpuSecond() - istart;
    printf("time: %f\n", ielpase);
    return 0;
}