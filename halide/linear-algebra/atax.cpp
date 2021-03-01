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
#define M 38 // 116 390 1900 1800
#define N 42 // 124 410 2100 2200

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
    int n=N, m=M;
    Buffer<DTYPE> A(m, n, "matrix A");
    Buffer<DTYPE> B(n, "vector B");
    Buffer<DTYPE> out(n, "output");
    srand(time(NULL));

    init_buffer(A, 2);
    init_buffer(B, 1);

    // Define Halide Vars
    Func atax("atax"), tmp("tmp");
    Var m("m"), k ("k");
    RDom r(0, n, "r");

    // Algorithm
    // out = (A * B) * (C * D)
    tmp(k) += A(k, r.x) * B(r.x);
    atax(k) += A(r.x, k) * tmp(r.x);

    // Using basic schedule...Can be optimized
    tmp.compute_root();
    atax.print_loop_nest();

    // Compile
    atax.compile_jit();

    // Execute or generate binary
    atax.realize(out);
    print_buffer(out, 1);
    return 0;
}
