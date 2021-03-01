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
#define NI 16 // 40 180 800 1600
#define NJ 18 // 50 190 900 1800
#define NK 22 // 70 210 1100 2200
#define NL 24 // 80 220 1200 2400

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
    int ni=NI, nj=NJ, nk=NK, nl=NL;
    Buffer<DTYPE> A(ni, nj, "matrix A");
    Buffer<DTYPE> B(nj, nk, "matrix B");
    Buffer<DTYPE> C(nk, nl, "matrix C");
    Buffer<DTYPE> D(ni, nl, "matrix D");
    Buffer<DTYPE> out(ni, nl, "output"); // Later check if this needed or D can be overwritten

    srand(time(NULL));

    init_buffer(A);
    init_buffer(B);
    init_buffer(C);
    init_buffer(D);
    DTYPE alpha, beta;
    // Initializing as defined in Polybench
    alpha = 1.5;
    beta = 1.2;

    // Define Halide Vars
    Func mm2("mm2"), tmp("tmp");
    Var m("m"), k ("k");
    RDom rj(0, nj, "nj");
    RDom rk(0, nk, "nk");

    // Algorithm
    tmp(m, k) = alpha * Halide::sum(A(m, rj.x) * B(rj.x, k));
    mm2(m, k) = beta * D(m, k);
    mm2(m, k) += tmp(m, rk.x) * C(rk.x, k);

    // Using basic schedule...Can be optimized
    tmp.compute_root();
    mm2.print_loop_nest();

    // Compile
    mm2.compile_jit();

    // Execute or generate binary
    mm2.realize(out);
    print_buffer(out);
    return 0;
}
