#include <cuda_runtime.h> 
#include <stdio.h>
#include "time_utils.h"

__global__ void warmup(float *c) {
    printf("warpSize: %d\n", warpSize);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;

    if ((tid % 2) == 0) {
        a = 100.0;
    }
    else {
        b = 200.0;
    }
    c[tid] = a + b;
}

__global__ void math_kernel1(float *c) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;

    if ((tid % 2) == 0) {
        a = 100.0;
    }
    else {
        b = 200.0;
    }
    c[tid] = a + b;
}

__global__ void math_kernel2(float *c) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;
    bool ipred = (tid % (2 * warpSize) == 0);

    
    //if ((tid / warpSize) % 2 == 0) {
    //    a = 100.0;
    //}
    if (ipred) {
        a = 100.0;
    }
    else {
        b = 200.0;
    }
    c[tid] = a + b;
}

__global__ void math_kernel3(float *c) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 0.0;
    float b = 0.0;

    bool ipred = (tid % 2 == 0);

    if (ipred) {
        a = 100.0;
    }
    else {
        b = 200.0;
    }
    c[tid] = a + b;
}

int main(int argc, char* argv[]) {
    int dev = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, device_prop.name);

    const int size = 64;
    const int k_thread_num = 256;

    dim3 block(k_thread_num, 1);
    printf("block.x: %d, block.y:%d\n", block.x, block.y);
    dim3 grid((size - 1) / block.x + 1, 1);

    size_t n_byte = size * sizeof(float);

    float *h_c = (float *) malloc(n_byte);
    float *d_c;
    cudaMalloc(&d_c, n_byte); 

    double i_start, i_elaps;
    cudaDeviceSynchronize();
    i_start = CpuSecond();
    warmup<<<grid, 1>>> (d_c);
    cudaDeviceSynchronize();
    i_elaps = CpuSecond() - i_start;

    printf("warmup  <<<%d, %d>>> elapsed %lf sec \n", grid.x, block.x, i_elaps);
    
    i_start = CpuSecond();
    math_kernel1<<<grid, block>>> (d_c);
    cudaDeviceSynchronize();
    i_elaps = CpuSecond() - i_start;

    printf("kernel1  <<<%d, %d>>> elapsed %lf sec \n", grid.x, block.x, i_elaps);

    cudaMemcpy(h_c, d_c, n_byte, cudaMemcpyDeviceToHost);
    /* 
    for (int i = 0; i < size; i++) {
        printf("%f ", h_c[i]);
    }
    printf("\n");
    */
    i_start = CpuSecond();
    math_kernel2<<<grid, block>>> (d_c);
    cudaDeviceSynchronize();
    i_elaps = CpuSecond() - i_start;

    printf("kernel2  <<<%d, %d>>> elapsed %lf sec \n", grid.x, block.x, i_elaps);

    i_start = CpuSecond();
    math_kernel3<<<grid, block>>> (d_c);
    cudaDeviceSynchronize();
    i_elaps = CpuSecond() - i_start;

    printf("kernel3  <<<%d, %d>>> elapsed %lf sec \n", grid.x, block.x, i_elaps);

    cudaFree(d_c);
    free(h_c);
    cudaDeviceReset();
    return 0;
}