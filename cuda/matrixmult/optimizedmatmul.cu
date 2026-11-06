#include <stdio.h>
#include <cuda_runtime.h>

#define TILESIZE 16

__global__ void matmul_tiled(float *C, const float *A, const float *B, int M, int N, int K){
   
    //Declaring arrays to store 16*16 potion of the A and B arrays in a_tile and b_tile respectively in shared memory
    __shared__ float b_tile[TILESIZE][TILESIZE];    //arrays stored in shared memory
    __shared__ float a_tile[TILESIZE][TILESIZE]; 
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILESIZE + ty;   //Determin the exact row of the thread in a block
    int col = blockIdx.x * TILESIZE + tx;   //Determin the exact column of the thread in a block
    
    float sum = 0.0f;

    //Load the values from 2D array into 1D memory allocation in shared memory
    for(int phase = 0; phase < (K + TILESIZE - 1) / TILESIZE; phase++){     //What is phase? Explanation down in the comment block
        
        //Loading the tiles from the global memory into shared memory
        a_tile[ty][tx] = 
            (row < M && phase * TILESIZE + tx < K)  
            ? A[row * K + phase * TILESIZE + tx]     //This similar to wriiting A[row][phase * TILESIZE + tx]
            : 0.0f;

        b_tile[ty][tx] = 
            (phase * TILESIZE + ty < K && col < N)  
            ? B[(phase * TILESIZE + ty) * N + col]      //This similar to wriiting A[phase * TILESIZE + ty][col]
            : 0.0f;
        
        //Sync
        __syncthreads();    //Wait until the function of all the threads are finished
        
        //Main computation of the function
        for(int i = 0; i < TILESIZE; i++){
            sum += a_tile[ty][i] * b_tile[i][tx];
        }
        __syncthreads();
    }

    if(row < M && col < N){
        C[row * N + col] = sum;
    }
}

int main(){

    const int M = 1024, N = 1024, K = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_a = (float*)malloc(bytes_A);
    float *h_b = (float*)malloc(bytes_B);
    float *h_c = (float*)malloc(bytes_C);

    for(int i = 0; i < M * K; i++){
        h_a[i] = 2.0f;
    }

    for(int i = 0; i < K * N; i++){
        h_b[i] = 3.0f;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes_A);
    cudaMalloc(&d_b, bytes_B);
    cudaMalloc(&d_c, bytes_C);

    cudaMemcpy(d_a, h_a, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_B, cudaMemcpyHostToDevice);

    //                      x         y
    dim3 threadsPerBlock(TILESIZE, TILESIZE);
    dim3 numBlocks((N + TILESIZE -1) / TILESIZE, (M + TILESIZE -1) / TILESIZE );

    matmul_tiled<<<numBlocks, threadsPerBlock>>>(d_c, d_a, d_b, M, N, K);

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess){
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_c, d_c, bytes_C, cudaMemcpyDeviceToHost);

    printf("This is the optimized version of the matrix multiplication program\n");
    printf("verification: c[0]=%.1f and c[1024*1024] = %.1f\n", h_c[0], h_c[(1024*1024)-1]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    free(h_a); free(h_b); free(h_c);

    return 0;
}



/*
//Level of abstrastion
    Grid -> Block -> Tiles -> Threads


//Anatomy of 1 tile
    +-------------------+   - Contains total of 16*16 = 256 threads
    |c0 c1 c2 ...    c15|   - The concept of tile seems to be equivalent to one block but differ during computation element size
    |r1                 |   - The program calculates one C element per thread   
    |r2                 |   - threadIdx.x and threadIdx.y describe the thread's position inside its block.
    |.                  |   - blockIdx.x and blockIdx.y describe the block's position inside the grid.
    |.                  |
    |r15                |
    +-------------------+

//Anatomy of a block
                                Block 0 - a_tile 
   +---------------------------------------------------------------------+ 
   | +-------------------+-------------------+     +-------------------+ |
   | |c0 c1 c2 ...    c15|c0 c1 c2 ...    c15|     |c0 c1 c2 ...    c15| |
   | |r1                 |r1                 |     |r1                 | |
   | |r2                 |r2                 |     |r2                 | |
   | |.     (TILE1)      |.                  | ... |.     (TILE15)     | |
   | |.                  |.                  |     |.                  | |
   | |r15                |r15                |     |r15                | |
   | +-------------------+-------------------+     +-------------------+ |
   | |c0 c1 c2 ...    c15|c0 c1 c2 ...    c15|                           |
   | |r1                 |r1                 |                           |
   | |r2                 |r2                 |                           |
   | |.                  |.                  |                           |
   | |.                  |.                  |                           |
   | |r15                |r15                |                           |
   | +-------------------+-------------------+                           |
   |         .                                                           | 
   |         .                                                           |
   |         .                                                           |
   | +-------------------+                                               |
   | |c0 c1 c2 ...    c15|                                               |
   | |r1                 |                                               | 
   | |r2                 |                                               |
   | |.     (TILE15)     |                                               |
   | |.                  |                                               |
   | |r15                |                                               |
   | +-------------------+                                               |
   +---------------------------------------------------------------------+ 

    - There will be also a b_tile for this block and both tiles will be sharing the same shared memory
    - For each block, these arrays (a_tile and b_tile) hold the input data needed to calculate its 16 × 16 output region of C.


Consider this example:
    blockIdx.y = 1;     //Block Row 1
    blockIdx.x = 2;     //Block Column 2

    threadIdx.y = 5;    //Thread Row 5
    threadIdx.x = 3;    //Thread Column 3

    row = 1(16)     + 5 = 21
    col = 2(16)     + 3 = 35
  blockIdx^ ^TILESIZE ^threadIdx

    The thread in row 21 and column 35 is resposible for calculating the value for C[21][35]

- Global memory belongs to the GPU and can be accessed by all thread blocks.
- Shared memory is allocated per thread block and can be accessed by all threads belonging to that block.
- Each thread also has its own private registers, where your sum will typically reside.
- So every block has its own a_tile and b_tile. Block 0 cannot directly access the shared-memory arrays belonging to block 1.
- For each block, these arrays hold the input data needed to calculate its 16 × 16 output region.


Why do we use phase?
- Remember that matrix multiplication involves summing over the K dimension.
- Your K dimension contains 1024 elements, but your shared-memory tile can only accommodate 16 elements along that dimension at a time.
- Therefore # of phases = 1024/16 = 64
- Each phase processes another 16 elements of the multiplication.
- Think of it as dividing the multiplication into 64 smaller multiplication operations.
- (K + TILESIZE - 1) / TILESIZE     //Way of performing integer division with rounding upwards. That's useful when K isn't an exact multiple of 16.
- So the K values are divided into 16 phases,
            phase       K
            0        0-15  
            1        16-31
            2        32-47
            .          .
            .          .
            .          .
            63       1008-1023              

    row * K + phase * TILESIZE + tx     //Similar to A[row][phase * TILESIZE + tx]
- row identifies the output row currently being processed.
- phase * 16 + tx identifies the current section of that row.
- So you're taking a portion of a row in A and placing it into shared memory.

    (phase * TILESIZE + ty) * N + col // Similar to B[phase * TILESIZE + ty][col]
- We're selecting a portion of a column range from B.
- phase * 16 + ty identifies the current section of rows.
- col identifies the output column being processed.

- This is important because matrix multiplication requires a row from A and a column from B.
- During phase 0, the block loads:
        A: 16 rows × 16 columns, corresponding to the block's output rows and K indices 0–15.
        B: 16 rows × 16 columns, corresponding to K indices 0–15 and the block's output columns.


__syncthreads()
- All 256 threads in a block load values into the shared-memory arrays. However, GPU threads don't necessarily finish their operations simultaneously
- Suppose thread 0 finishes loading its values, but thread 200 hasn't finished
- Thread 0 must not start reading the shared-memory tile before thread 200 has finished writing its portion.
- Otherwise, it could read an incomplete tile.
- __syncthreads() acts as a block-wide barrier where every thread in the block must reach this point before any of them proceed beyond it.




*/