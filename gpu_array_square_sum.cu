#include <stdio.h>
#define THREAD_NUM 256 

const int data_size = 256000;

void GenerateNumbers(int *data) {
    for(int i = 0; i < data_size; i++) {
        data[i] = rand() % 10;
    }
}

__global__ void square_1(int *d_in, int *d_out) {
    int idx = threadIdx.x;
    int f = d_in[idx];
    d_out[idx] = f * f;
}

__global__ void SumOfSquare(int *din, int *d_out, clock_t *duration) {
    const int idx = threadIdx.x;
    int size = data_size / THREAD_NUM; 
    int sum = 0;
    int i;
    clock_t start;
    if (idx == 0) start = clock();
    for (i = idx * size; i < (idx + 1) * size; i++) {
        sum += din[i] * din[i];
    }
    d_out[idx] += sum;
    if(idx == 0) *duration = clock() - start;
}

void CpuAdd(int *din) {
    clock_t start = clock();
    int sum = 0;
    for (int i = 0; i < data_size; i++) {
        sum += din[i] * din[i];
    }
    clock_t end = clock();
    printf("cpu sum: %d, time:%d \n", sum,  end - start);

}

int main() {
    const int data_byte = data_size * sizeof(int);
    const int res_byte = THREAD_NUM * sizeof(int);
    int h_in[data_size];
    int h_out[THREAD_NUM];
    GenerateNumbers(h_in);

    int *d_in;
    int *d_out;
    clock_t *d_duration;
    clock_t h_duration;

    cudaMalloc((void **) &d_in, data_byte);
    cudaMalloc((void **) &d_out, res_byte);
    cudaMalloc((void **) &d_duration, sizeof(clock_t));
    // 从Host拷贝到Device
    cudaMemcpy(d_in, h_in, data_byte, cudaMemcpyHostToDevice);

    // 1个kernel，ARRAY_SIZE 线程
    SumOfSquare<<<1, THREAD_NUM>>>(d_in, d_out, d_duration);

    // 从Device拷贝到Host
    cudaMemcpy(h_out, d_out, res_byte, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_duration, d_duration, sizeof(clock_t), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_out, d_out, res_byte, cudaMemcpyDeviceToHost);
    int final_num = 0;
    for (int i = 0; i < THREAD_NUM; i++) {
        final_num += h_out[i];
    }
    printf("gpu add %d\n", final_num);
    printf("time:%d\n", h_duration);

    CpuAdd(h_in);
    
    // 释放内存
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_duration);
    return 0;
}