#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define HADDLE_ERROR(err) HaddleError(err, __FILE__, __LINE__)


void HaddleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d \n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

void InitImage(float *image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image[i * width + j] =  (i + j) % 256;
        }
    }
}

void VisualKernel(float *arr, int width, int height) {
    for (int i = 0; i < height; i++)  {
        for (int j = 0; j < width; j++) {
            printf("%2.f ", arr[i * width + j]);
        }
        printf("\n");
    }
}

void VisualImage(float *arr, int width, int height, int visual_size) {
    for (int i = 0; i < visual_size; i++)  {
        for (int j = 0; j < visual_size; j++) {
            printf("%2.f ", arr[i * width + j]);
        }
        printf("\n");
    }
}

__global__ void warmup(float *image, float *kernel, float *res,
                     int width, int height, int kernel_size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x; 
    int id = bid * blockDim.x + tid;

    if (id >= width * height) return;

    int row = id / width;
    int col = id % width;

    int cur_row, cur_col;
    float res_val = 0;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            cur_row = row - kernel_size / 2 + i;
            cur_col = col - kernel_size / 2 + j;

            if (cur_row < 0 || cur_col < 0 || cur_row > height|| cur_col > width) {
                res_val = 0; 
            } else {
                res_val = image[cur_row * width + cur_col];
            }
            res[id] += res_val * kernel[i * kernel_size + j];
        }
    }
}

__global__ void conv(float *image, float *kernel, float *res,
                     int width, int height, int kernel_size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x; 
    int id = bid * blockDim.x + tid;

    if (id >= width * height) return;

    int row = id / width;
    int col = id % width;

    int cur_row, cur_col;
    float res_val = 0;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            cur_row = row - kernel_size / 2 + i;
            cur_col = col - kernel_size / 2 + j;

            res_val = 0;
            if (cur_row > 0 && cur_col > 0 && cur_row < height && cur_col < width) {
                res_val = image[cur_row * width + cur_col];
            }
            res[id] += res_val * kernel[i * kernel_size + j];
        }
    }
}

__global__ void conv1(float *image, float *kernel, float *res,
                     int width, int height, int kernel_size) {
    int tid = threadIdx.x;
    int bid = blockIdx.x; 
    int id = bid * blockDim.x + tid;

    if (id >= width * height) return;

    int row = id / width;
    int col = id % width;

    int cur_row, cur_col;
    float res_val = 0;
    for (int i = 0; i < kernel_size; i++) {
        cur_row = row - kernel_size / 2 + i;
        cur_col = col - kernel_size / 2 + 0;
        res_val = 0; 
        if (cur_row > 0 && cur_col > 0 && cur_row < height && cur_col < width) {
            res_val = image[cur_row * width + cur_col];
        }
        res[id] += res_val * kernel[i * kernel_size + 0];

        cur_col = col - kernel_size / 2 + 1;
        res_val = 0; 
        if (cur_row > 0 && cur_col > 0 && cur_row < height && cur_col < width) {
            res_val = image[cur_row * width + cur_col];
        }
        res[id] += res_val * kernel[i * kernel_size + 1];

        cur_col = col - kernel_size / 2 + 2;
        res_val = 0; 
        if (cur_row > 0 && cur_col > 0 && cur_row < height && cur_col < width) {
            res_val = image[cur_row * width + cur_col];
        }
        res[id] += res_val * kernel[i * kernel_size + 2];
    }
}

int main(int argc, char* argv[])  {
    srand((int) time(0));

    int width = 1920;
    int height = 1080;
    int img_size = width * height;

    int thread_nums = 256;
    if (argc > 1) {
        thread_nums = atoi(argv[1]);
    }
    int block_nums = (img_size - 1) / thread_nums + 1;

    float *img_host = new float[img_size];
    InitImage(img_host, width, height);
    float *res_host = new float[img_size];

    int kernel_width = 3;
    int kernel_height = 3;
    int kernel_size = kernel_width * kernel_height;
    float *kernel_host = new float[kernel_size];
    for (int i = 0; i < kernel_height; i++) {
        for (int j = 0; j < kernel_width; j++) {
            kernel_host[i * kernel_width + j] =  j % kernel_width - 1;
        }
    }

    float *kernel_dev;
    float *img_dev;
    float *res_dev;
    HADDLE_ERROR(cudaMalloc((void **)&kernel_dev, sizeof(float) * kernel_size));
    HADDLE_ERROR(cudaMalloc((void **)&img_dev, sizeof(float) * img_size));
    HADDLE_ERROR(cudaMalloc((void **)&res_dev, sizeof(float) * img_size));

    HADDLE_ERROR(cudaMemcpy(img_dev, img_host, sizeof(float) * img_size, cudaMemcpyHostToDevice));
    HADDLE_ERROR(cudaMemcpy(kernel_dev, kernel_host, sizeof(float) * kernel_size, cudaMemcpyHostToDevice));

    // image visualization
    VisualImage(img_host, width, height, 10);
    // kernel visualization
    VisualKernel(kernel_host, kernel_width, kernel_height);
    // warm up
    dim3 block(thread_nums, 1);
    dim3 grid(block_nums, 1);
    warmup<<<grid, block>>> (img_dev, kernel_dev, res_dev, width, height, kernel_width);
    cudaDeviceSynchronize();

    conv<<<grid, block>>> (img_dev, kernel_dev, res_dev, width, height, kernel_width);
    cudaDeviceSynchronize();

    conv1<<<grid, block>>> (img_dev, kernel_dev, res_dev, width, height, kernel_width);
    cudaDeviceSynchronize();
    // process image visualization
    HADDLE_ERROR(cudaMemcpy(res_host, res_dev, sizeof(float) * img_size, cudaMemcpyDeviceToHost));
    VisualImage(res_host, width, height, 10);

    cudaFree(kernel_dev);
    cudaFree(img_dev);
    cudaFree(res_dev);
    return 0;
}