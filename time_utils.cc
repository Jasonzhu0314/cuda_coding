#include <sys/time.h>
#include <iostream>
#include "time_utils.h"


double CpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ( (double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}