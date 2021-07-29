#include <stdio.h>
#include <iostream>
#include <vector>

using namespace std;

void MatrixCalculate(int *m, int *n, int *p, int w, int h) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < h; j++) {
            int sum = 0;
            for (int k = 0; k < w; k++) {
                sum += m[i * w + k] * n[k * h + j];
            }
            p[i * h + j] = sum;
        }
    }
} 

void PrintMatrix(int *m, int w, int h) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            cout << m[i * w + j] << "\t";
        }
        cout << endl;
    }
}

int main() {
    const int w = 5;
    const int h = 2;
    int *m = new int[w * h];
    int *n = new int[h * w];
    int *p = new int[h * h];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            m[i * w + j] = i * j;
        }
    }
    printf("m array:\n");
    PrintMatrix(m, w, h);

    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            n[i * h + j] = i * j;
        }
    }
    printf("n array:\n");
    PrintMatrix(n, h, w);

    printf("result array:\n");
    MatrixCalculate(m, n, p, w, h);
    PrintMatrix(p, h, h);

    return 0;
}