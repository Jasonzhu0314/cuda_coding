#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
using namespace std;

// 1.函数外部定义，不能初始化，要在线程中初始化，不同的block中会创建一个备份，不共享，都有shared memory
__shared__ int shared_val1;

__global__ void test_shared_val() {
    // 2.函数内部定义数组 不能初始化，要在线程中初始化
    __shared__ int shared_val[10];
    // 3.函数内部定义变量 不能初始化，要在线程中初始化
    __shared__ int shared_val2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        shared_val1 = 10;
        shared_val2 = 20;
        shared_val[0] = 30;
    }

    // 用于数据同步，否则会出现其他线程共享内存的数据没有赋值的现象，打印结果0
    __syncthreads();
    printf("blockid: %d, tid: %d, shared_val is %s\n", blockIdx.x, idx, __isShared(&shared_val) ? "shared": "not shared");
    // block 0 和 block 1打印的结果不一样
    printf("blockid: %d, tid: %d, shared_val1: %d\n", blockIdx.x,  idx, shared_val1);
    printf("blockid: %d, tid: %d, shared_val2: %d\n", blockIdx.x, idx, shared_val2);
    printf("blockid: %d, tid: %d, shared_val: %d\n", blockIdx.x, idx, shared_val[0]);
    // TODO: cout 
    //cout << "shared_val:" << shared_val[0] << endl;
}

__global__ void test_shared_val1() {
    // 4. 调用kernel时，定义shared mem空间大小 
    extern __shared__ int shared_val[];
    if (threadIdx.x == 0) {
        shared_val[0] = 30;
    }

    __syncthreads();

    printf("extern tid: %d, shared_val: %d\n",threadIdx.x ,shared_val[0]);
}

int main() {
    const int block_nums = 2;
    const int thread_nums = 2;

    test_shared_val<<<block_nums, thread_nums>>> ();

    // <<<block_nums, thread_num, sharead_memory_size>>> 参数，每一个kernel调用时都创建一个sharedmemory
    int shared_mem_size = sizeof(int) * 5;
    test_shared_val1<<<block_nums, thread_nums, shared_mem_size>>> ();

    printf("test_shared_val\n");
    cudaDeviceSynchronize();
    return 0;
}
