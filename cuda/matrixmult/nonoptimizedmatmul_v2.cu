//Compile using the nvcc compiler
#include <stdio.h>
#include <cuda_runtime.h>

//A[M][K] * B[K][N] = C[M][N]
__global__ void matmult(float *C, const float *A, const float *B, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;    //Rows: defined in y dir, think of in sequential memory
    int col = blockIdx.x * blockDim.x + threadIdx.x;    //Columns: defined in X dir
    if(row < M && col < N){  //Setting boundry conditioons so that the program does not access memory out side the allocated size
        float sum = 0.0f;
        for(int k = 0; k < K; k++){     //The only loop that runs within a thread
            sum += A[row * K + k] * B[k * N + col]; 
        }
        C[row * N + col] = sum;
    }
}

int main(){
    const int M = 1024, N = 1024, K =1024;
    size_t bytes_a = M * K * sizeof(float); //Size of the whole array A
    size_t bytes_b = K * N * sizeof(float);
    size_t bytes_c = M * N * sizeof(float);
   
    float *h_a = (float *)malloc(bytes_a);
    float *h_b = (float *)malloc(bytes_b);
    float *h_c = (float *)malloc(bytes_c);

    //filling the arrays with data
    for(int i = 0; i < M * K; i++){
        h_a[i] = 1.0f;
    }
    for(int j = 0; j < K * N; j++){
        h_b[j] = 2.0f;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    //Copy the values from the CPU memory to GPU memory
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

    //For 2D             (X,  Y)
    dim3 threadsPerBlock(16, 16);   //IF it were 1D we would ave used 256, but here as we are working in 2D, so 16 for X and 16 for Y (Total (256) = 16 * 16)
    dim3 numBlocks((N + 15)/16, (M + 15)/16);   //As the X dir represented by N and Y dir represented by M

    matmult<<<numBlocks, threadsPerBlock>>>(d_c, d_a, d_b, M, N, K);

    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

    //Confirming the results
    printf("Matrix Multiplication without any optimizations:\n");
    printf("Verification: C[0] = %.1f and c[1024*1024]= %.1f\n", h_c[0], h_c[(1024*1024) - 1]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}
/*
- Notice that the matmul function only has one loop and our CPU implementation had 3?
- The main apparent difference here is we are not using the array format here, rather we are using seqeuncial memory form the heap instead of the memory from the stack in the CPU implementation
- So we wont have the first two "for" loops that iterate thrrough the 2D array
- Then we spread out the work to individual threads such that we can work in parallel instaed of the sequential method used in the CPU implementation
- The only "for" loop in the function works within one thread to calculate the values for C

- There is a glaring problem here as the threads move on from one calculation of C to the next they have to goto global memory and fetch the values for the calculation again and again
- This process is relatively slow, so what we can do is something like locally caching in the CPU, we get some common values and store it in a local shared memory
- This memory is only shared within on block, although the size of this shared memory is relatively small to the global memory, fetching data from it much faster

- In the algorithm of matrix multiplication we can do another optimization by breaking down and sharing the amount of calculations done by one thread
- For this we introduce a abstraction called tilling, where we can group serveral of the threads withhin a block called tile to calculate only one element in the output matric C
- So one element of matrix C will be calculated by 2 tiles (or matrix A and matrix B). Tiles in the optimizedimplementation will be 2D arrays in shared memory
- So suppose we have 2 matrixes of size A[1024][1024] and B[1024][1024], now consider matrix A only we have 1024*1024 elements in it and same goes to matrix B
- Now suppose we want to multiply the teo matrixes to form matrix C[1024][1024], so now when we want to calculate the value of each element of C we have to choose the corrensponding row on A as 
same as C and the corrensponding column on B as same as C, multiply and add all the values to find the value of C, so we will do
                C(0,0) = A(0,0)B(0,0) + A(0,1)B(1,0)...A(0,1023)B(1023,0)
everytime to find the value of one element in C, here is the problem we talked earlier comes as this a massive number of data being processed at one time accessing time to global memory is noticable (This is for only one imagine for all the elements in the C matrix)

- So what we do is split or tile the matrixes, A[1024][1024] => 64{ a_tile[16][16] }
- Here we shrunk the size of the unit of processing into 16*16, so now we can load the 16*16 into shared memory and process a_tile and b_tiles serparately spreading out the load through out the threads, then afterward we add the values from the two tiles to get final value for one position in the C matrix 
- Now each thread processes only 256*256 calculations rather than 1024*1024 for each element in C


*/