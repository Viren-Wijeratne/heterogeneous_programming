#include <stdio.h>
#define M 2
#define N 3
#define P 2

//Simple algorithm for matrix multiplication
void matmult(int A[M][N], int B[N][P], int C[M][P]){
    for(int i = 0; i < M; i++){     //Goes through the rows of the matrix C
        for(int j = 0; j < P; j++){     //Goes through the columns of the matrix C
            C[i][j] = 0;    //This neccessary as when we go to the next position after running the inner most loop we need to reset the next element of C, to be 0
            for(int k = 0; k < N; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
            
        }
    }

}

int main(){

    int A[M][N] = {
        {1, 2, 3},
        {4, 5, 6}
    };

    int B[N][P] = {
        {7, 8},
        {9, 10},
        {11, 12}
    };

    int C[M][P] = {0};

    matmult(A, B, C);
 
    for(int p = 0; p < M; p++){
        for(int q = 0; q < P; q++){
            printf("%2d ", C[p][q]);
        }
        printf("\n");
    }

    return 0;
}