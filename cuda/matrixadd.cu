#include <stdio.h>
#include <cuda_runtime.h>

//Goal: out[X][Y][Z] = a[X][Y][Z] + b[X][Y] + c[X]
//Matrix add in 3D: Z * Width * Height + Y * Width + X
__global__ void matadd(float *out, const float *a, const float *b, const float *c, int X, int Y, int Z){
    int row = blockIdx.y * blockDim.y + threadIdx.y;    //y-coordinate of the current thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;    //x-coordinate of the current thread
    int dep = blockIdx.z * blockDim.z + threadIdx.z;    //z-coordinate of the current thread

    if(row < Y && col < X && dep < Z){ 
        out[dep * X * Y + row * X + col] = a[dep * X * Y + row * X + col] + b[row * X + col] + c[col];   
    }
}

int main(){
    const int X = 10, Y = 10, Z = 10;
    size_t bytes_a = X * Y * Z * sizeof(float); //Size of the whole array A
    size_t bytes_b = X * Y * sizeof(float);
    size_t bytes_c = X * sizeof(float);
    size_t bytes_out = X * Y * Z * sizeof(float);
   
    float *h_a = (float *)malloc(bytes_a);
    float *h_b = (float *)malloc(bytes_b);
    float *h_c = (float *)malloc(bytes_c);
    float *h_out = (float *)malloc(bytes_out);

    //filling the arrays with data
    for(int i = 0; i < X * Y * Z; i++){
        h_a[i] = 1.0f;
    }
    for(int j = 0; j < X * Y; j++){
        h_b[j] = 2.0f;
    }
    for(int j = 0; j < X ; j++){
        h_c[j] = 2.0f;
    }

    float *d_a, *d_b, *d_c, *d_out;

    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);
    cudaMalloc(&d_out, bytes_out);

    //Copy the values from the CPU memory to GPU memory
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes_c, cudaMemcpyHostToDevice);

    //For 2D             (X,  Y, Z)
    dim3 threadsPerBlock(8, 8, 8);   // Cant use 16*16*16 = 4096, as the CUDA block can contain a maximum of 1024 threads (on typical NVIDIA architectures). 
    dim3 numBlocks((X + 7)/8, (Y + 7)/8, (Z + 7)/8);   //As the X dir represented by N and Y dir represented by M

    matadd<<<numBlocks, threadsPerBlock>>>(d_out, d_a, d_b, d_c, X, Y, Z);

    cudaMemcpy(h_out, d_out, bytes_out, cudaMemcpyDeviceToHost);

    //Confirming the results
    printf("Matrix addition of out[X][Y][Z] = a[X][Y][Z] + b[X][Y] + c[X] :\n");
    printf("Verification: out[0] = %.1f and out[10*10*10]= %.1f\n", h_out[0], h_out[(10*10*10) - 1]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_out);
    free(h_a); free(h_b); free(h_c); free(h_out);

    return 0;
}

/* 
- When the Z plane of the matrix is taken out of the picture the matrix becomes a 2D problem
- Think of this problem as tissue papers stack on top of each other, and we want to get to the middle tissue papers middle rows, middle column

X => Number of elements in an array row
Y => Number of rows in one plane
Z => Number of planes
col => X coordinate of the current thread
row	=> Y coordinate of the current thread
hei	=> Z coordinate of the current thread


*/