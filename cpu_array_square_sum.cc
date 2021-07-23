#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>
#include <mutex>
#include <unistd.h>

#define DATA_MULTI 102400
#define THREAD_NUM 256

const unsigned int cpu_thread_num = 2;
const int data_size = 102400000;
int data[data_size];
int res = 0;
std::mutex res_lock;

void GenerateNumbers() {
    for(int i = 0; i < data_size; i++) {
        data[i] = rand() % 10;
    }
}

void CpuSumOfSquare(int thread_no) {
    int nums_per_thread = data_size / cpu_thread_num;
    int start = thread_no * nums_per_thread;
    int end = (thread_no + 1) * nums_per_thread;
    int sum = 0;
    for (int i = thread_no;i < data_size; i += cpu_thread_num) {
        sum += data[i] * data[i]; 
    }
    res_lock.lock();
    res += sum;
    res_lock.unlock();
} 


int main(int argc, char* argv[]) {
    
    GenerateNumbers();
    std::thread *thread_groups[cpu_thread_num];

    clock_t start = clock();

    for(int i = 0; i < cpu_thread_num; i++) {
        std::thread *t = new std::thread(CpuSumOfSquare, i);
        thread_groups[i] = t;
    }

    for (int i = 0; i < cpu_thread_num; i++) {
        thread_groups[i]->join();
    }
    clock_t duration = clock() - start;


    printf("data_size: %d , cpu thread: %d, time: %f, res: %ld\n",
            data_size, cpu_thread_num, (double) duration / CLOCKS_PER_SEC, res);


    return 0;
}
