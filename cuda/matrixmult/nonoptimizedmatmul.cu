//Got too complicated, program not finished!!!

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define M 3
#define N 4
#define P 3


//A[M][N] * B[N][P] = C[M][P]
__global__ void matmult(int A[M][N], int B[N][P], int C[M][P]){
    int row = blockIdx.y * blockDim.y + threadIdx.y;    //Rows: defined in y dir, think of in sequential memory
    int col = blockIdx.x * blockDim.x + threadIdx.x;    //Columns: defined in X dir
    if(row < M && col < N){  //Setting boundry conditioons so that the program does not access memory out side the allocated size
        for(int j = 0; j < N; j++){
            float sum = 0.0f;
            for(int k = 0; k < P; k++){
                //sum += A[row * K + k] * B[k * N + col]; 
                //C[row * N + col] = sum;
                C[row * P][col] += A[row * N][k] * B[N * k][col];       //?????
            }
    }
}
}

int main(){
    size_t bytes_a = M * N * sizeof(float); //Size of the whole array A
    size_t bytes_b = N * P * sizeof(float);
    size_t bytes_c = M * N * sizeof(float);
   
    float h_a[M][N] = malloc(bytes_a);      //Wrong implementation!!!
    float h_b[N][P] = (float)malloc(bytes_b);
    float h_c[M][P] = (float)malloc(bytes_c);

    //filling the arrays with data
    for(int i = 0; i < M * N; i++){
        h_a[i] = 1.0f;
    }
    for(int j = 0; j < N * P; j++){
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