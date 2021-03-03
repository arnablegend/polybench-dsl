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
#define NI 20 // 60 200 1000 2000
#define NJ 25 // 70 220 1100 2300
#define NK 30 // 80 240 1200 2600

void print_buffer(Buffer<DTYPE> buf){
    for(int j = buf.min(1); j < buf.extent(1); j++){
        for(int i = buf.min(0); i < buf.extent(0); i++)
            cout << buf(i, j) << " ";
        cout << endl;
    }
}

void init_buffer(Buffer<DTYPE> buf){
    for(int j = buf.min(1); j < buf.extent(1); j++)
        for(int i = buf.min(0); i < buf.extent(0); i++)
            buf(i, j) = (DTYPE)((rand() % 100) / 10.0 + 1);
}

int main(int argc, char** argv){
    int ni=NI, nj=NJ, nk=NK;
    Buffer<DTYPE> A(nk, ni, "matrix A");
    Buffer<DTYPE> B(nj, nk, "matrix B");
    Buffer<DTYPE> C(nj, ni, "matrix C");
    Buffer<DTYPE> out(nj, ni, "matrix C");

    srand(time(NULL));

    init_buffer(A);
    init_buffer(B);
    DTYPE alpha, beta;
    // Initializing as defined in Polybench
    alpha = 1.5;
    beta = 1.2;

    // Define Halide Vars
    Func gemm("gemm"), tmp("tmp");
    Var m("i"), k ("j");
    RDom rk(0, nk, "nk");

    // Algorithm
    tmp(j, i) = alpha * Halide::sum(A(rk.x, i) * B(j, rk.x));
    gemm(j, i) = tmp(j, i) + beta * C(j, i);

    // Using basic schedule...Can be optimized
    tmp.compute_root();
    gemm.print_loop_nest();

    // Compile
    gemm.compile_jit();

    // Execute or generate binary
    gemm.realize(out);
    print_buffer(out);
    return 0;
}
