#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
using namespace std;

// 1.函数外部定义，不能初始化，要在线程中初始化
__shared__ int shared_val1;

__global__ void test_shared_val() {
    // 2.函数内部定义数组 不能初始化，要在线程中初始化
    __shared__ int shared_val[10];
    // 3.函数内部定义变量 不能初始化，要在线程中初始化
    __shared__ int shared_val2;

    if (threadIdx.x == 0) {
        shared_val1 = 10;
        shared_val2 = 20;
        shared_val[0] = 30;
    }

    // 用于数据同步，否则会出现其他线程共享内存的数据没有赋值的现象，打印结果0
    __syncthreads();
    printf("shared_val is %s\n",__isShared(&shared_val) ? "shared": "not shared");

    printf("tid: %d, shared_val1: %d\n", threadIdx.x, shared_val1);
    printf("tid: %d, shared_val2: %d\n", threadIdx.x, shared_val2);
    printf("tid: %d, shared_val: %d\n",threadIdx.x, shared_val[0]);
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

    test_shared_val<<<1, 2>>> ();

    // <<<block_nums, thread_num, sharead_memory_size>>> 参数
    test_shared_val1<<<1, 2, sizeof(int) * 5>>> ();

    printf("test_shared_val\n");
    cudaDeviceSynchronize();
    return 0;
}
