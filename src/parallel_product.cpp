#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <sycl/ext/intel/fpga_extensions.hpp>

void d_norm(sycl::queue &q, double *vec, double *ans, int N) {
  double sum = 0;
  double *temp = malloc_shared<double>(N,q);
  q.submit([&](sycl::handler& h){
    h.parallel_for(N,[=](sycl::id<1> i){
      temp[i] = vec[i]*vec[i];
    });
  });

  q.wait();

  for(int j=0;j<N;j++){sum += temp[j];}

  *ans = sqrt(sum);

  free(temp,q);
}

void d_vec_minus_mat_vec(sycl::queue &q, double *val, int *col_ind, int *row_ptr, double *x, double *b, double *ans, int N) {
  double *temp = malloc_shared<double>(N,q);
  q.submit([&](sycl::handler& h){
    h.parallel_for(N,[=](sycl::id<1> i){
    for(int j=row_ptr[i]-1; j<row_ptr[i+1]-1;j++){
      temp[i] += val[j] * x[col_ind[j]-1];
      }
    });
  });
    
  q.wait();
    
  q.submit([&](sycl::handler& h){
    h.parallel_for(N,[=](sycl::id<1> i){
      ans[i] = b[i] - temp[i];
    });
  });
    
  q.wait();

  free(temp,q);
}

void d_vec_div(sycl::queue &q, double *vec, double *c, int N) {
  q.submit([&](sycl::handler& h){
    h.parallel_for(N,[=](sycl::id<1> i){
      vec[i] = vec[i] / *c;
    });
  });

  q.wait();
}

void d_mat_vec(sycl::queue &q, double *val, int *col_ind, int *row_ptr, double *x, double *y, int N) {
  q.submit([&](sycl::handler& h){
    h.parallel_for(N,[=](sycl::id<1> i){
    for(int j=row_ptr[i]-1; j<row_ptr[i+1]-1;j++){
      y[i] += val[j] * x[col_ind[j]-1];
      }
    });
  });

  q.wait();
}

void d_inner_product(sycl::queue &q, double *a_vec, double *b_vec, double *ans, int N) {
  double *temp = malloc_shared<double>(N,q);
  q.submit([&](sycl::handler& h){
    h.parallel_for(N,[=](sycl::id<1> i){
      temp[i] = a_vec[i] * b_vec[i];
    });
  });

  q.wait();

  for(int j=0;j<N;j++){*ans += temp[j];}

}

void d_vec_minus_cons_vec(sycl::queue &q, double *a_vec, double *b_vec, double *cons, int N) {
  q.submit([&](sycl::handler& h){
    h.parallel_for(N,[=](sycl::id<1> i){
      a_vec[i] -= *cons * b_vec[i];
    });
  });

  q.wait();
}

void d_backward_substitution(sycl::queue &q, double *U, double *y, double *b, int n, int istep) {
    int i, j;
    double temp;
    for(i=0; i<n; i++) {
        temp = b[n-1-i];
        for(j=0; j<i; j++) {
            temp -= U[(n-1-j)*(istep+1)+n-1-i] * y[n-1-j];
        }
        y[n-1-i] = temp / U[(n-1-i)*(istep+1)+n-1-i];
    }
}
