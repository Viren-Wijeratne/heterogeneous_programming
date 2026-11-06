#include <stdio.h>
#include <cuda_runtime.h>

//Functions running in the GPU cannot return anything, results  of any calculations done within the gpu is stored in its memory which should be copied back to the cpu memory inorder to use it
//Host = CPU, Device = GPU
//Definition of a function to be run within the gpu's global memory space is defined using __global__ flag

__global__ void addKernel(int *c, const int *a, const int *b, int size){	//Functions in the GPU are called kernel functions
	int i = blockIdx.x * blockDim.x + threadIdx.x;	//blockId = Block index, blockDim = Block Dimension (# threads per block), threadIdx = Thread Index within its block, gives a unique 1D index to each thread
	if(i < size){c[i] = a[i] + b[i];}	//check if the size goes beyond the calculated value i (which denotes the boundary of a memory space in the gpu). This is neccessary as we will be allocating threads in chunks of 32 or 1 warp
}

int main(){

	const int size = 1000000, bytes = size * sizeof(int);

	int *h_a = (int*)malloc(bytes);		//Memory allocation within the CPU memory
	int *h_b = (int*)malloc(bytes);
	int *h_c = (int*)malloc(bytes);

	for(int i = 0; i < size; i++){		//Populating the variables
		h_a[i] = i;
		h_b[i] = i * 2;
	}

	int *dev_a, *dev_b, *dev_c;

	cudaMalloc(&dev_a, bytes);		//Memory allocation of variables within the GPU
	cudaMalloc(&dev_b, bytes);
	cudaMalloc(&dev_c, bytes);

	cudaMemcpy(dev_a, h_a, bytes, cudaMemcpyHostToDevice);	//Copying over the data in the CPU memory into the GPU memory
	cudaMemcpy(dev_b, h_b, bytes, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;			//256 threads make 8 warps per block, number of threads inside 1 block
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;		//Number of blocks we want to run

	//Inorder to run our kernel we need to specific how may blocks we want to run and how many threads are in one block
	addKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b, size);	//Call the function to run within the GPU

	cudaMemcpy(h_c, dev_c, bytes, cudaMemcpyDeviceToHost);		//After the function ran the result will be stored at dev_c in the GPU memory, Here we copy it back into the CPU memory

	printf("Vector addition of %d elements\n", size);
	printf("Verification: c[0]=%d, c[99999]=%d\n", h_c[0], h_c[99999]);

	cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);	//Free all allocated memory within the GPU
	free(h_a); free(h_b); free(h_c);			//Free all allocated memory within the CPU
	return 0;

}

/*
- threadsPerBlock = 256 is a chosen launch configuration, not a value derived from size. 
- It’s a common starting point because CUDA executes threads in groups of 32 called warps, so 256 threads make 8 warps per block. 
- It’s not automatically the fastest choice for every kernel or GPU; you can benchmark other values. 

- int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock; 
- For size = 1,000,000, it gives 3,907 blocks, or (3907 * 256) 1,000,192 threads total. The extra 192 threads get indices beyond the array’s end (array ends at thread 1,000,000), so if (i < size) prevents out-of-bounds access. 
- CUDA organizes a kernel launch as a grid of blocks, with threads in each block.
 */
