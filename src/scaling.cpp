#include <iostream>
#include <math.h>
#include <sycl/ext/intel/fpga_extensions.hpp>

void scaling(double* d_val, int* row_ptr, double* d_b, int N, int param) {
    int i, j;
    double max;
    double pow_2_param = 1;
    for (i = 0; i < param; i++) {
        pow_2_param = pow_2_param * 2;
    }
    for (i = 0; i < N; i++) {
        max = 0;
        for (j = row_ptr[i] - 1; j < row_ptr[i + 1] - 1; j++) {
            if (fabs(d_val[j]) > max) {
                max = fabs(d_val[j]);
            }
        }
        if (max != 0) {
            for (j = row_ptr[i] - 1; j < row_ptr[i + 1] - 1; j++) {
                d_val[j] = d_val[j] * pow_2_param / max;
            }
            d_b[i] = d_b[i] * pow_2_param / max;
        }
    }
}

