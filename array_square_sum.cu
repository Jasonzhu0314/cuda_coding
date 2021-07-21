#include <stdio.h>

__global__ void square_1(float *d_in, float *d_out) {
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f * f;
}



int main() {
    const int ARRAY_SIZE = 8;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);    

    // h前缀一般表示host， d前缀一般表示device
    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];

    float* d_in;
    float* d_out;

    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // 从Host拷贝到Device
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // 1个kernel，ARRAY_SIZE 线程
    square_1<<<1, ARRAY_SIZE>>>(d_in, d_out);

    // 从Device拷贝到Host
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%f\n", h_out[i]);
    }
    
    // 释放内存
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
