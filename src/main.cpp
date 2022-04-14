#include <CL/sycl.hpp>
#include <iostream>
#include <string>
#include <math.h>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../include/read_mm.hpp"
#include "../include/scaling.hpp"
#include "../include/parallel_product.hpp"
#include "../include/gmres.hpp"

using namespace sycl;

constexpr int scaling_param = 16;
constexpr int decimal_bit = 30;

int main(int argc, char *argv[]) {

#if FPGA_EMULATOR
 ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  ext::intel::fpga_selector d_selector;
#else
  default_selector d_selector;
#endif

    queue q(d_selector);
    
    // Print out the device information used for the kernel code.
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "#onaAPI-double-GMRES computing starts" << std::endl;

    int N, NNZ;
    int i;
    std::cout << "#scaling_param " << scaling_param << std::endl;
    std::cout << "#decimal_bit " << decimal_bit << std::endl;

    // Reading the matrix data
    double *d_val;
    int *col_ind;
    int *row_ptr;
    char filename[100];
    const char *filedir = "../matrix/"; 
    strcpy(filename, filedir);
    strcat(filename, argv[1]);
    if(read_mm(filename, &d_val, &col_ind, &row_ptr, &N, &NNZ, q) == 1) return 1;

    std::cout << "#row_ptr is: ";
    for(i=0;i<8;i++){
        std::cout << row_ptr[i] << " ";
    }
    std::cout << ". . ." << std::endl;

    std::cout << "#col_ind is: ";
    for(i=0;i<8;i++){
        std::cout << col_ind[i] << " ";
    }
    std::cout << ". . ." << std::endl;

    // initialization of the right-hand side vector
    double *d_b = malloc_shared<double>(N,q);
    double *d_original_b = malloc_shared<double>(N,q);
    
    for(i=0; i<N; i++) {
        d_b[i] = 1;
    }

    scaling(d_val, row_ptr, d_b, N, scaling_param);
    for(i=0; i<N; i++) {
        d_original_b[i] = d_b[i];
    }

    std::cout << "#After scaling, val is: ";
    for(i=0;i<8;i++){
        std::cout << d_val[i] << " ";
    }
    std::cout << ". . ." << std::endl;

    //Execution of GMRES in parallel
    double *ans = malloc_shared<double>(N,q);
    q.submit([&](sycl::handler& h){
      h.parallel_for(N,[=](sycl::id<1> i){
        ans[i] = 0;
      });
    });
    q.wait();
    double_gmres(q, d_val, col_ind, row_ptr, d_b, d_original_b, ans, N);

    free(d_val,q);
    free(col_ind,q);
    free(row_ptr,q);
    free(ans,q);
    free(d_b,q);
    free(d_original_b,q);
}
