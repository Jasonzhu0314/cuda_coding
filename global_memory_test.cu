#include <stdio.h>

#define NUM 10
#define BLOCK_DIM 2

// 方法一：外部使用__device__申请
__device__ float *device_s_array;

__global__ void global_memory_test(float *arr) {
    // global memory 不同的block之间都可以访问
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //int idx = threadIdx.x;
    extern __shared__ float shared_val[BLOCK_DIM * NUM];
    shared_val[idx] = idx;
    arr[idx] = shared_val[idx];
    //device_s_array[idx] = 1;
}

int main() {

    float *host_s_array;
    float *host_array;
    float *device_array;
    host_s_array = (float *) malloc(BLOCK_DIM * sizeof(float) * NUM);
    host_array = (float *) malloc(BLOCK_DIM * sizeof(float) * NUM);
    cudaMallocHost(&device_array, BLOCK_DIM * sizeof(float) * NUM);
    //cudaMallocHost(&device_s_array, BLOCK_DIM * sizeof(float) * NUM);

    global_memory_test <<<BLOCK_DIM, NUM, BLOCK_DIM * sizeof(float) * NUM>>> (device_array);

    cudaDeviceSynchronize();
    
    cudaMemcpy(host_array, device_array, BLOCK_DIM * sizeof(float) * NUM, cudaMemcpyDeviceToHost);
    //cudaMemcpy(host_s_array, device_s_array, BLOCK_DIM * sizeof(float) * NUM, cudaMemcpyDeviceToHost);

    printf("host_array\n");
    for (int i = 0; i < BLOCK_DIM * NUM; i++) {
        printf("host[%d]:%f\n", i, host_array[i]);
    }

    //printf("host_s_array\n");
    //for (int i = 0; i < NUM; i++) {
    //    printf("host[%d]:%f\n", i, host_s_array[i]);
    //}

    cudaFreeHost(device_array);
    //cudaFreeHost(device_s_array);

    return 0;
}
