#ifndef READ_MM_HPP
#define READ_MM_HPP

using namespace sycl;

int read_mm(char *filename, double **val, int **col_ind, int **row_ptr, int *N, int *NNZ, queue &q);

#endif
