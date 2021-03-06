# oneAPI-GMRES
DPC++ double GMRES, can run on CPU GPU or and FPGA emulator. But cannot run on the FPGA device, due to the memory allocation problem.

Document description:
All files are rewritten in C/C++ mixed style now, and more C++/DPC++ features may be used in the future.
1. The parallel_product.cpp program uses Basic-Parallel kernel wrote by myself, it's possible to use oneMKL library in the future. And NDrange kernels are more suitable in FPGA which pipeline is efficient.
2. The read_mm.cpp can read matrix file information which is as same as CPU version except for memory allcation type. The scaling.cpp is also identity with CPU version, just remain the double version.
3. The gmres.cpp can execute the function in the parallel_product.cpp file in parallel. The main.cpp organizes all other files.

Command line steps:
(CMake or Makefile still in progress)
Compile and run this program in GPU:
1. qsub -I -l nodes=1:gpu:ppn=2 -d .
2. dpcpp main.cpp read_mm.cpp gmres.cpp scaling.cpp parallel_product.cpp -o gpurun
3. ./gpurun wang3.mtx

Compile and run this program in FPGA Emulator:
1. qsub -I -l nodes=1:fpga_compile:ppn=2 -d .
2. dpcpp -fintelfpga main.cpp read_mm.cpp gmres.cpp scaling.cpp parallel_product.cpp -o fpgaemu -DFPGA_EMULATOR=1
3. qsub -I -l nodes=1:fpga_runtime:arria10:ppn=2 -d . (this command can be ignored)
4. ./fpgaemu wang3.mtx

Attention: following FPGA hardware's way is aborted, just to refer in the future.

Compile and run this program in FPGA hardware:
It is a little different from emulator, running on the hardware needs batch job.
1. create script file job.sh like:
#!/bin/bash
dpcpp -fintelfpga main.cpp read_mm.cpp gmres.cpp scaling.cpp parallel_product.cpp -o fpga -Xshardware -DFPGA=1
2. submit job.sh to the compile nodes:
qsub -l nodes=1:fpga_compile:ppn=2 -d . job.sh -l walltime=24:00:00
3. run on the execution nodes:
qsub -I -l nodes=1:fpga_runtime:arria10:ppn=2 -d . (or stratix10)
4. ./fpga wang3.mtx

Here, If it cannot run the executable, changing computing nodes in devcloud maybe useful.
Input following commands: 
1. source /data/intel_fpga/devcloudLoginToolSetup.sh
2. devcloud_login
