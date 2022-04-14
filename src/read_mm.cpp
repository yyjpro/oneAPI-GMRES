#include <CL/sycl.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sycl/ext/intel/fpga_extensions.hpp>

#define general 0
#define symmetric 1

static void swap(int *a, int *b) {
	int temp = *a;
	*a = *b;
	*b = temp;
}

static void d_swap(double *a, double *b) {
	double temp = *a;
	*a = *b;
	*b = temp;
}

static void quick_sort(int *row_ptr, int *col_ind, double *val, int left, int right) {
	int Left, Right;
	int pivot;
	Left = left; Right = right;
	pivot = row_ptr[(left + right) / 2];
	while (1) {
		while (row_ptr[Left] < pivot) Left++;
		while (pivot < row_ptr[Right]) Right--; 
		if (Left >= Right) break;
		swap(&row_ptr[Left], &row_ptr[Right]);
		swap(&col_ind[Left], &col_ind[Right]);
		d_swap(&val[Left], &val[Right]);
		Left++; Right--;
}
	if (left < Left - 1) quick_sort(row_ptr, col_ind, val, left, Left - 1);
	if (Right + 1 < right) quick_sort(row_ptr, col_ind, val, Right + 1, right);
}
//double **val = &d_val: val stores d_val(point) address
int read_mm(char *filename, double **val, int **col_ind, int **row_ptr, int *N, int *NNZ, sycl::queue &q) {
	int i, j, type, PE;
	FILE *fp;
	char firstline[100];
	if( (fp = fopen(filename, "r")) == NULL ) {
		std::cout << "read_matrix: reading matrix error" << std::endl;
		return 1; //filename = "./matrix/wang3.mtx"
	}
	fgets(firstline, 100, fp);//Read the FP file word by word to the array firstline, with a maximum of 100
	if(strcmp(firstline, "%%MatrixMarket matrix coordinate real general\n") == 0) {
		type = general;//string compare, success, return 0
	}else if(strcmp(firstline, "%%MatrixMarket matrix coordinate integer general\n") == 0) {
		type = general;
	}else if(strcmp(firstline, "%%MatrixMarket matrix coordinate real symmetric\n") == 0) {
		type = symmetric;
	}else {
		std::cout << "read_matrix: MM type or format error" << std::endl;
		return 1;
	}
	while(fgetc(fp) == '%') {
        while(fgetc(fp) != '\n');
    }//int fgetc(FILE *stream):Read a character. After reading a byte, move the cursor position back by one byte
    fseek(fp,-1,SEEK_CUR);//get current position of the cursor and move it back one bit
	fscanf(fp, "%d %d %d", N, N, &PE);//Read the value from the current position, stored into N N PE
	while(fgetc(fp) != '\n');//After reading the first line, move the cursor position to the front of the first element in the second line
	std::cout << "#N " << *N << std::endl;
	std::cout << "#pattern entries " << PE << std::endl;
	if(type == general) {
		*NNZ = PE;//PE: nonzeros(NNZ) = 177168
		std::cout << "#type genaral" << std::endl;
	}else if(type == symmetric) {
		*NNZ = PE + PE - *N;
		std::cout << "#type symmetric" << std::endl;
	}
    //int *temp_row_ptr = (int *)malloc(sizeof(int)*(*NNZ));
	//*val = (double *)malloc(sizeof(double)*(*NNZ));
	//*col_ind = (int *)malloc(sizeof(int)*(*NNZ)); 
	//*row_ptr = (int *)malloc(sizeof(int)*(*N+1));
    int *temp_row_ptr = malloc_shared<int>(*NNZ,q);
	*val = malloc_shared<double>(*NNZ, q);
    *col_ind = malloc_shared<int>(*NNZ, q);
    *row_ptr = malloc_shared<int>(*N+1, q);
	double scan_val;
	int scan_col;
	int scan_row;
	int real_NNZ=0;//The value of the real non-zero element is initially set to 0
	if(type == general) {
		for(i=0; i<PE; i++) {
        	fscanf(fp,"%d %d %lf",&scan_row, &scan_col, &scan_val);//first column is the value of row
        	while(fgetc(fp) != '\n');
        	if(scan_val == 0) {
       			if(scan_row == scan_col) {
					std::cout << "read_matrix: scan error" << std::endl;
       				return 1;
       			}
        	}else if(scan_val != 0) {
        		temp_row_ptr[real_NNZ] = scan_row;//Each array starts storing data from real_NNZ = 0
        		(*col_ind)[real_NNZ] = scan_col;
        		(*val)[real_NNZ] = scan_val;
        		real_NNZ++;//Non zero number increases by one
        	}
    	}
    	*NNZ = real_NNZ; // remove 0 from NNZ
    	quick_sort(temp_row_ptr, *col_ind, *val, 0, *NNZ-1);//quick sort based on row number
		std::cout << "quicksort successful!" << std::endl;
	}else if(type == symmetric) {
		for(i=0; i<PE; i++) {
        	fscanf(fp,"%d %d %lf",&scan_row, &scan_col, &scan_val);
        	while(fgetc(fp) != '\n');
        	if(scan_val == 0) {
       			if(scan_row == scan_col) {
					std::cout << "read_matrix: scan error" << std::endl;
       				return 1;
       			}
        	}else if(scan_val != 0) {
        		temp_row_ptr[real_NNZ] = scan_row;
        		(*col_ind)[real_NNZ] = scan_col;
        		(*val)[real_NNZ] = scan_val;
        		real_NNZ++;
        		if(scan_row != scan_col) {
        			temp_row_ptr[real_NNZ] = scan_col;
        			(*col_ind)[real_NNZ] = scan_row;
        			(*val)[real_NNZ] = scan_val;
        			real_NNZ++;
        		}
        	}
    	}
    	*NNZ = real_NNZ; // remove 0 from NNZ
    	quick_sort(temp_row_ptr, *col_ind, *val, 0, *NNZ-1);
	}
    int row_num = 1;
    int count = 1, count_prev = 1;
    (*row_ptr)[0] = 1;
    for(i=0; i<*NNZ; i++) {
    	if(temp_row_ptr[i] == row_num) {
    		count++;
    	}else {
    		(*row_ptr)[row_num] = count; 
    		quick_sort(*col_ind, temp_row_ptr, *val, count_prev-1, count-2);
    		row_num++;
    		count++;
    		count_prev = count-1;
    	}
    }
    (*row_ptr)[*N] = count;
    quick_sort(*col_ind, temp_row_ptr, *val, count_prev-1, count-2);
	if(*val == NULL || *col_ind == NULL || *row_ptr == NULL) {
		std::cout << "read_matrix: malloc error" << std::endl;
		return 1;
	}
	free(temp_row_ptr,q);
	return 0;
}