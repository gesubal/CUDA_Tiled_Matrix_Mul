//Parallel programming for many core GPUs
//Name: Gesu Bal
//Instructor name: Meilin Liu
/*
this is a simple cuda program calculating Tiled Matrix vector multiplication for 2 dimensions on GPU device
I multiplied two double two-dimensional matrices A, B on the device GPU. 
After the device matrix multiplication kernel function is invoked, and the multiplication result is transferred back to the CPU. 
The program will also compute the  multiplication matrix of matrices A and B using the CPU.  
Then the program compares the device-computed result with the CPU-computed result. 
If it matches (within a certain tolerance, i.e., 0.000001), then it will print out "Test PASSED" to the screen before exiting.

This case is for all matrix sizes and blocksize/tilewidth: 8*8
*/

#include<stdio.h>
#include<cuda.h>
#include <time.h>
int N,blocksize;

//gpu function for multiplication
__global__ void mul_matrix_gpu(double *d_a, double *d_b, double *d_c, int width)
{  
    int TILE_WIDTH=8;
    __shared__ double ds_M[8][8];
    __shared__ double ds_N[8][8];

     int bx = blockIdx.x;  int by = blockIdx.y;
     int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the Pd element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    double Pvalue = 0;
    // Loop over the Md and Nd tiles required to compute the Pd element
    for (int m = 0; m < width/TILE_WIDTH; m++) 
    {
    // Coolaborative loading of Md and Nd tiles into shared memory
        ds_M[ty][tx] = d_a[Row*width + m*TILE_WIDTH +tx];
        ds_N[ty][tx] = d_b[Col+(m*TILE_WIDTH+ty)*width];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
             Pvalue += ds_M[ty][k] * ds_N[k][tx];
        __syncthreads();
    }   
     d_c[Row*width+Col] = Pvalue;
	 
	 
}

//cpu function for multiplication
void mul_matrix_cpu(double *a, double *b, double *cpu_c, int N)
{ 
int i, j,k; 
for (i=0;i<N;i++) { 
       for (j=0;j<N;j++) {
             double sum=0;
                for (k=0;k<N;k++) 
                {
                 double p=a[i*N+k];
                 double q=b[k*N+j];
                 sum=sum+(p*q);
                }
	    cpu_c[i*N+j]=sum;
     } 
  } 
} 

//cpu and gpu result matching function
bool verify(double *A, double *B, double *C, int  width) {
     const double relativeTolerance = 0.000001; 
     for(int row = 0; row < width; row++) {
        for(int col = 0; col < width; col++) {
            double sum = 0;
            for(unsigned int k = 0; k < width; k++) {
                    sum += A[row*width + k]*B[k*width + col];
            }
            double relativeError = (sum - C[row*width + col])/sum;
	    //printf("%f \t",relativeError);
	    //printf("\n");
            if (relativeError >= relativeTolerance
             || relativeError <= -relativeTolerance) 
            {
                    printf("TEST FAILED\n\n");
                    return false;
            }
           }
        }
      printf("TEST PASSED\n\n");
      return true; 
}

//print matrix
int printMatrix(double *a,int N)
{
  int i,j;
  for (i=0;i<N;i++)
	{
		for (j=0;j<N;j++)
		{
		  printf("%f\t",a[i*N+j]);
		}
		printf("\n");
	}
return 1;
  
}


int main()
{
    //user input
  
    int r, col;
	printf("Select one of the following options: \n");
	printf("Press a for matrix size 8 * 8 \n");
	printf("Press b for matrix size 64 * 64 \n");
	printf("Press c for matrix size 128 * 128 \n");
	printf("Press d for matrix size 512 * 512 \n");
	printf("Press e for matrix size 1024 * 1024 \n");
	printf("Press f for matrix size 4096 * 4096 \n");
	printf("Press any other key for exit \n");
	char ch;
	scanf("%c",&ch);
	switch(ch)
        {
            case 'a':
                r=8;
		col=8;
		N=8;
		printf("Matrix size is 8 * 8 \n");
		
		break;
            case 'b':
                r=64;
		col=64;
		N=64;
		
		printf("Matrix size is 64 * 64 \n");
		
		break;
            case 'c':
                r=128;
		col=128;
		N=128;
		
		printf("Matrix size is 128 * 128 \n");
		
		break;
	    case 'd':
                r=512;
		col=512;
		N=512;
		
		printf("Matrix size is 512 * 512 \n");
		
		break;
		
            case 'e':
                r=1024;
		col=1024;
		N=1024;
		
		printf("Matrix size is 1024 * 1024 \n");
		break;
		
	    case 'f':
		r=4096;
		col=4096;
		N=4096;
		
		printf("Matrix size is 4096 * 4096 \n");
		break;
	    
	    default:
		exit(1);
                break;            
	} 
	
	//initializing the block size/tile width
	blocksize=8;
       
	//memory allocation for vectors
	double *a, *b, *c, *cpu_c, *d_a, *d_b, *d_c;
	
	int a_size=r*col;
	int b_size=r*col;
	int c_size=r*col;
	int cpu_c_size=r*col;
		
	a=(double*)malloc(sizeof(double)*a_size);
	b=(double*)malloc(sizeof(double)*b_size);
	c=(double*)malloc(sizeof(double)*c_size);
	cpu_c=(double*)malloc(sizeof(double)*cpu_c_size);
		
	
	//matrix initialization
	int i,j;
	int init=1325;
	for (i=0;i<N;i++)
	{
		for (j=0;j<N;j++)
		{
		    init=3125*init%65536;
		    a[i*col+j]=(init-32768.0)/16384.0;
		    init=3125*init%65536;
		    b[i*col+j]=(init-32768.0)/16384.0;
		}
	}
	//printMatrix(a,N);
	//printf("\n");
	//printMatrix(b,N);
	//printf("\n");
	
	//allocating memory on device
	int cudaret=cudaMalloc((void **)(&d_a),(N*N)*sizeof(double));
	if(cudaret!=cudaSuccess)
	{printf("memory was not allocated on device \n");}
	
	cudaMalloc((void **)(&d_b),(N*N)*sizeof(double));
	cudaMalloc((void **)(&d_c),(N*N)*sizeof(double));
	
	//calculating cpu time
	clock_t startCPU, end;
	float cpu_time_used;
	
	//calling CPU program
	printf("Calculating results for CPU vector multiplication \n");
	printf("---------\n");
	startCPU = clock();
	mul_matrix_cpu(a,b,cpu_c,N);
	end = clock();
	cpu_time_used = ((float) (end - startCPU))*1000;
	cpu_time_used= cpu_time_used/ CLOCKS_PER_SEC;
	
	printf("CPU computation time (milliseconds) \n");
	printf("%f \t",cpu_time_used);  
	printf("\n");
	printf("\n");
	//printMatrix(cpu_c,N);
	//printf("\n");
	//time execution calculation
	cudaEvent_t start,stop;
	float elapsedTime; 
	float timeTransfer;
	float timeBack;
   
	
	
	//memory transfer time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	//copying contents of a and b to device arrays
	cudaMemcpy(d_a,a,(N*N)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,(N*N)*sizeof(double),cudaMemcpyHostToDevice);
	
	cudaDeviceSynchronize(); 
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeTransfer,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	
	
	//Initializing block count and block size
	dim3 dimBlock(blocksize,blocksize,1); 
	int blockCount_x = (N - 1)/(double(blocksize))+1;//Get number of blocks needed per direction.
	int blockCount_y = (N - 1)/(double(blocksize))+1;
	
	dim3 dimGrid(blockCount_x,blockCount_y,1);
	printf("Block size and tile width for the program is %d\n ",blocksize);
	//call kernel for gpu functioning
	printf("Calling kernel for gpu computations for vector multiplication and calculating results\n");
	printf("---------\n");
    
    
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	mul_matrix_gpu<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,N);
	cudaEventRecord(stop,0);
     	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	printf("GPU computation time (milliseconds) \n");
	printf("%f \t",elapsedTime);  
	printf("\n");
    
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	//copying resulting back to cpu
	cudaMemcpy(c,d_c,(N*N)*sizeof(double),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize(); 
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeBack,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	timeTransfer += timeBack;
	printf("Total Memory transfer time between CPU and GPU (milliseconds)\n");
	printf("%f \t",timeTransfer);  
	printf("\n");
	
	float speedup;
	speedup=cpu_time_used/elapsedTime;
	printf("Speedup: \n");
	printf("%f \t",speedup);  
	printf("\n");

	printf("Comparing results for CPU and GPU computations \n");
	printf("---------\n");
	verify(a,b,c,N);
	    
	//deallocating memory 
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
    return 0;
}
