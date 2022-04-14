#ifndef GMRES_HPP
#define GMRES_HPP

using namespace sycl;

void double_gmres(queue &q, double *d_val, int *col_ind, int *row_ptr, double *d_b, double *d_original_b, double *ans, int N);

#endif