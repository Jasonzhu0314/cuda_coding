#include <stdio.h>
#include <cuda_runtime.h>

bool InitCuda() {
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        printf("This is no device.\n");
        return false;
    }
    printf("This is %d device.\n", count);

    return true;
}

int main() {
    bool temp = InitCuda();
    return 0;
}
