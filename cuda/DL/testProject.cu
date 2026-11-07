//Neural Network to identify hand written numbers from 0 - 9
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

//This is a matrix multiplication for the forward pass
//The operation is calculating the dot product between the inputs and the weight added to the bias input[].weight[]+bias
//Further we can stack weights into a matrix and the biases into a vector to get an equation for the ouptut of every neuron
//Usually we calculate the output for serveral inputs, which in turn the inputs can be stacked into a matrix as well
//varibales batch_size, n and out_w are used to define the size of the matrices
__global__ void forward(int batch_size, int n, int out_w, float *input, float *weights, float *biases, float *output){
    int col = blockIdx.x * blockDim.x + threadIdx.x;    //x-cordinate of the kernel grid
    int row = blockIdx.y * blockIdx.y + threadIdx.y;    //y-cordinate of the kernel grid

    if(row < batch_size && col < out_w){
        output[row * out_w + col] = biases[col];    //Initalizing the output to the corresponding bias
        for(int i = 0; i < n; i++){
            output[row * out_w + col] += weights[i * out_w + col] * input[row * n + i]; //We iterate over the weights and inputs and calculate the dot product 
        }
    }
}

//We need this activation function to tell the network what threshold is needed for the element to pass as valid value
//Non-linear activation funnction - ReLU activation
//This activation kernel will be used for every layer in the network except for the last one, as we will be outputting the probabilty of an image representing each number from 0 to 9 
__global__ void relu(int w, int h, float *input, float *output){
    int col = blockIdx.x * blockDim.x + threadIdx.x;    //x-cordinate of the kernel grid
    int row = blockIdx.y * blockIdx.y + threadIdx.y;    //y-cordinate of the kernel grid
    if(row< h && col < w){
        //if the activation > 0 output[row * w + col] = activation; or if < 0 output[row * w + col] = 0.0 
        float activation = input[row * w + col];
        output[row * w + col] = activation > 0.f ? activation : 0.f;
        //                                    ^Here we set the activation threshold  
    }
}

//We need a kernel that would take the in our final layers output and return a probability distribution for a number it thinks is the input
//For this we will use the softmax function as our final activation (We didnt use ReLU in the last layer because of this)
//As the softmax function is an exponential function that grows exponentially, if out input vector contains multiple positive values it can overflow as we will be adding alot of big numbers together in our divisor 
//We can mitigate this by substracting the maximum of our vector from the exponent, there by the values will always be negative and our values will be in the range of 0 to 1
__global__ void softmax(int w, int h, float *input, float *output){
    int col = blockIdx.x * blockDim.x + threadIdx.x;    //x-cordinate of the kernel grid
    int row = blockIdx.y * blockIdx.y + threadIdx.y;    //y-cordinate of the kernel grid
    if(row < h && col < w){
        float maxval = input[row * w];  //Initialize the max value to be the first element of the input vector
        for(int i = 1; i < w; i++){
            maxval = max(maxval, input[row * w + i]);   //Then we iterate over the elements to find the maximum value
        }
        float divisor = 0.f;
        for(int i = 0; i < w; i++){
            divisor += exp(input[row * w + i] - maxval); //Iterate over the value to calculate the divisor (denominator of the softmax function)
        }
    output[row * w + col] = exp(input[row * w + col] - maxval) / (divisor); //Calculate the probabilty for our element and store it in the output array
    }
}

//A loss function is used to gauge how bad the neutral network is performing
//Higher the loss function worse is the result of our output
//For this we will use a Cross Entropy Loss function
//The functions takes two input vectors, one a vector that represents the real probability for each class and the other vector of predictions of our neural network
//The function iterates through all labels; numebers from  0-9 and then calculates the the cross entropy between the truth and our prediction
//Our real probability will have one entry that is eqaul to 1 indicating the real number and rest of the vector will be all 0s, 
//The functions output tends to go to 0 when the neural network has correctly guessed the label, tends to exponentially grow whne the guess is incorrect compared to the true value
__global__ void cross_entropy(int w, int h, float *preds, float *real, float *output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  //Only operate on the X axis of the kernel grid 
    if(idx < h){
        float loss = 0.f;
        for(int i = 0; i < w; i++){     //Iterate through each number and calculate its contribution to our loss 
            loss -= real[idx * w + i] * log(max(1e-6, preds[idx * w + i]));    //We do the max function with a very small number for numerical stability because the log function is undefined for zero
        }
        output[idx] = loss;
    }
}

//This function initializes the weight to be some random value
__global__ void init_rand(int w, int h, float *mat){
    int col = blockIdx.x * blockDim.x + threadIdx.x;    //x-cordinate of the kernel grid
    int row = blockIdx.y * blockIdx.y + threadIdx.y;    //y-cordinate of the kernel grid
    if(row < h && col < w){
        curandState state;  //from curand_kernel.h
        curand_init(42, row * w + col, 0, &state);
        mat[row * w + col] = curand_normal(&state) * sqrt(2.f/h);   //Squart root function is used here for the key initialization
    }
}


int main(){


    return 0;
}