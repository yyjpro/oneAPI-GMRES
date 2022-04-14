#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../include/parallel_product.hpp"

constexpr double oE = 1e-8;
constexpr int istep = 10;

void double_gmres(sycl::queue &q, double *d_val, int *col_ind, int *row_ptr, double *d_b, double *d_original_b, double *ans, int N){

    int j;
    double tmp;
    double d_norm_b;
    double d_norm_original_b;

    double *d_x = malloc_shared<double>(N,q);
    double *y = malloc_shared<double>(istep,q);
    double *c = malloc_shared<double>(istep,q);
    double *s = malloc_shared<double>(istep,q);
    double *e = malloc_shared<double>(istep+1,q);
    double *H = malloc_shared<double>(istep*(istep+1),q);
    double *V = malloc_shared<double>(N*(istep+1),q);;
    
    for(int i=0;i<N;i++){d_x[i]=0;}

    for(int i=0; i<istep; i++) {y[i] = 0;}
    for(int i=0; i<istep; i++) {c[i] = 0;}
    for(int i=0; i<istep; i++) {s[i] = 0;}

    for(int i=0;i<istep*(istep+1);i++){H[i]=0;}
    
    for(int i=0;i<N*(istep+1);i++){V[i]=0;}
    
    d_norm(q, d_b, &d_norm_original_b, N);
    std::cout << "#d_norm_original_b is: " << d_norm_original_b << std::endl;
    
    d_vec_minus_mat_vec(q, d_val, col_ind, row_ptr, d_x, d_b, &V[0], N);

    d_norm(q, &V[0], &e[0], N);
    std::cout << "#e[0] is: " << e[0] << std::endl;

    d_vec_div(q, &V[0], &e[0], N);
    
    for(j=0;j<istep;j++) {
      d_mat_vec(q, d_val, col_ind, row_ptr, &V[j*N], &V[(j+1)*N], N);
      for(int i=0; i<=j; i++){
        d_inner_product(q, &V[i*N], &V[(j+1)*N], &H[j*(istep+1)+i], N);
        d_vec_minus_cons_vec(q, &V[(j+1)*N], &V[i*N], &H[j*(istep+1)+i], N);
      }
      d_norm(q, &V[(j+1)*N], &H[j*(istep+1)+j+1], N);
      d_vec_div(q, &V[(j+1)*N], &H[j*(istep+1)+j+1], N);
      double temp;
      for(int i=0; i<j; i++) {
        temp = H[j*(istep+1)+i];
        H[j*(istep+1)+i] = c[i] * temp - s[i] * H[j*(istep+1)+(i+1)];
        H[j*(istep+1)+i+1] = s[i] * temp + c[i] * H[j*(istep+1)+(i+1)];
      }
      tmp = H[j*(istep+1)+j] * H[j*(istep+1)+j] + H[j*(istep+1)+j+1] * H[j*(istep+1)+j+1];
      tmp = sqrt(tmp);
      c[j] = H[j*(istep+1)+j] / tmp;
      s[j] = -H[j*(istep+1)+j+1] / tmp;
      e[j+1] = s[j] * e[j];
      e[j] = c[j] * e[j];
      H[j*(istep+1)+j] = tmp;
      H[j*(istep+1)+j+1] = 0;
    }
    d_backward_substitution(q, H, y, e, j, istep);
    for(int i=0; i<j; i++) {
      q.submit([&](sycl::handler& h){
        h.parallel_for(N,[=](sycl::id<1> l){
          d_x[l] = d_x[l] + y[i] * V[i*N+l];
        });
      });
      q.wait();
    }
    for(int i=0; i<N; i++) {
        ans[i] += d_x[i];
    }
    d_vec_minus_mat_vec(q, d_val, col_ind, row_ptr, ans, d_original_b, d_b, N);

    d_norm(q, d_b, &d_norm_b, N);
    std::cout << "#istep is: " << istep << std::endl;
    std::cout << "#After parallel computing, d_norm_b is: " << d_norm_b << ", d_norm_original_b is: " 
    << d_norm_original_b << ", so d_norm_b/d_norm_original_b is: " << d_norm_b/d_norm_original_b << std::endl;
    std::cout << "#End of onaAPI-double-GMRES computing" << std::endl;
    if (d_norm_b < oE * d_norm_original_b) {
        std::cout << "The minimum error has been reached, and the program exits" << std::endl;
        std::cout << "-------------------------------------------------------------" << std::endl;
        exit(0);
    }else{std::cout << "-------------------------------------------------------------" << std::endl;}
    
    free(d_x,q);
    free(y,q);
    free(c,q);
    free(s,q);
    free(e,q);
    free(H,q);
    free(V,q);
}