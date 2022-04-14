#ifndef PRODUCT_HPP
#define PRODUCT_HPP

using namespace sycl;

void d_norm(queue &q, double *vec, double *ans, int N);
void d_vec_minus_mat_vec(sycl::queue &q, double *val, int *col_ind, int *row_ptr, double *x, double *b, double *ans, int N);
void d_vec_div(sycl::queue &q, double *vec, double *c, int N);
void d_mat_vec(sycl::queue &q, double *val, int *col_ind, int *row_ptr, double *x, double *y, int N);
void d_inner_product(sycl::queue &q, double *a_vec, double *b_vec, double *ans, int N);
void d_vec_minus_cons_vec(sycl::queue &q, double *a_vec, double *b_vec, double *cons, int N);
void d_backward_substitution(sycl::queue &q, double *U, double *y, double *b, int n, int istep);

#endif