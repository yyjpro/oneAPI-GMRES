#!/bin/bash
dpcpp -fintelfpga main.cpp read_mm.cpp gmres.cpp scaling.cpp parallel_product.cpp -o fpga -Xsharedware -DFPGA=1