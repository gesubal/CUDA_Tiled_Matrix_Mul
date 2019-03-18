# CUDA_Tiled_Matrix_Mul
System requirements: CUDA 7.5 environment

Path setup for running cuda programs:
These two commands need to be executed in console for setting up the CUDA environment path.
$export PATH=/usr/local/cuda-7.5/bin:$PATH

$export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

Path setup can be confirmed using command:
	
	nvcc --version

The output should be like this:

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2015 NVIDIA Corporation
Built on Tue_Aug_11_14:27:32_CDT_2015
Cuda compilation tools, release 7.5, V7.5.17

Note: Vector types are Double to avoid floating point imprecision and lower precision for float

Task: Tiled Matrix Multiplication for two dimensional vectors:

Description: 
For this task, I developed a complete CUDA program which is divided into cases resulting into four files:
Case 1: matrix size 8 * 8 and block size 4 * 4 (this case is created because no other matrix size requires block size of 4 * 4 according to project specification)
  File name: "GesuVecMul_case1.cu"

Case 2: matrix size can be selected by user (8 * 8, 64 * 64, 128 * 128, 512 * 512, 1024 * 1024, 4096 * 4096) and block size: 8 * 8
  File name: "GesuVecMul_case2.cu"
  
Case 3: matrix size can be selected by user and block size: 16 * 16
  File name: "GesuVecMul_case3.cu"
  
Case 4: matrix size 4096 * 4096 and block size 32 * 32 (this case is created because no other matrix size requires block size of 32 * 32 according to project specification)
  File name: "GesuVecMul_case4.cu"

Program compilation:
For program compilation we need to be in same folder where all the files are present which can done using "cd /path/" command.
program can be compiled using the make file "Makefile" and typing the following command in terminal:

    make

Program Execution:
After successful compilation, program can be executed using command:

For case 1: 
./vec1
For case 2: 
./vec2
For case 3:
./vec3
For case 4: 
./vec4

On execution: program would ask user to enter the array size for the vector from a list for case 2 and case 3:
Enter a if you want 8*8 matrix
Enter b if you want 64*64 matrix
Enter c if you want 128*128 matrix
Enter d if you want 512*512 matrix
Enter e if you want 1024*1024 matrix
Enter f if you want 4096*4096 matrix

After entering the letter, program would calculate multiplication results for both GPU and CPU and then compare them and display "test passed" if both the results match.
Also program would calculate CPU computation time, GPU computation time and memory transfer time between CPU and GPU. 
