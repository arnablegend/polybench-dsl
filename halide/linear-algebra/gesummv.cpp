#include "Halide.h"
#include <cstdio>
#include "iostream"
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace Halide;

// choose your data type
#define DTYPE float

// using the mini-dataset, other datasets are commented, use whichever desired
#define N 30 // 90 250 1300 2800

void print_buffer(Buffer<DTYPE> buf, int num_dims){
    if(num_dims == 1){
        for(int i = buf.min(0); i < buf.extent(0); i++)
            cout << buf(i) << " ";
    }else if(num_dims == 2){
        for(int j = buf.min(1); j < buf.extent(1); j++){
            for(int i = buf.min(0); i < buf.extent(0); i++)
                cout << buf(i, j) << " ";
            cout << endl;
        }
    }
}

void init_buffer(Buffer<DTYPE> buf, int num_dims){
    if(num_dims == 1){
        for(int i = buf.min(0); i < buf.extent(0); i++)
            buf(i) = (DTYPE)((rand() % 100) / 10.0 + 1);
    }else if(num_dims = 2){
        for(int j = buf.min(1); j < buf.extent(1); j++)
            for(int i = buf.min(0); i < buf.extent(0); i++)
                buf(i, j) = (DTYPE)((rand() % 100) / 10.0 + 1);
    }
}

int main(int argc, char** argv){
    int n=N;
    Buffer<DTYPE> A(n, n, "matrix A");
    Buffer<DTYPE> B(n, n, "matrix A");
    Buffer<DTYPE> x(n, "vector x");
    Buffer<DTYPE> y(n, "vector y");
    srand(time(NULL));

    init_buffer(A, 2);
    init_buffer(B, 2);
    init_buffer(x, 1);
    DTYPE alpha = 1.5, beta = 1.2;

    // Define Halide Vars
    Func f_y("x1"), tmp("x2");
    Var i("i"), j("j");
    RDom r(0, n, "r");

    // Algorithm
    // out = (A * B) * (C * D)
    tmp(i) += A(r.x, i) * x(i);
    f_y(i) += B(r.x, i) * x(i);
    f_y(i) = alpha * tmp(i) + beta * y(i);

    // Using basic schedule...Can be optimized
    tmp.compute_root();
    f_y.print_loop_nest();

    // Compile
    f_y.compile_jit();

    // Execute or generate binary
    f_y.realize(y);
    print_buffer(y, 1);
    return 0;
}
