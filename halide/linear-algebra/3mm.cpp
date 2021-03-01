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
#define NK 20 // 60 200 1000 2000
#define NL 22 // 70 210 1100 2200
#define NM 24 // 80 220 1200 2400

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
    int ni=NI, nj=NJ, nk=NK, nl=NL, nm = NM;
    Buffer<DTYPE> A(ni, nk, "matrix A");
    Buffer<DTYPE> B(nk, nj, "matrix B");
    Buffer<DTYPE> C(nj, nm, "matrix C");
    Buffer<DTYPE> D(nm, nl, "matrix D");
    Buffer<DTYPE> out(ni, nl, "output");
    srand(time(NULL));

    init_buffer(A);
    init_buffer(B);
    init_buffer(C);
    init_buffer(D);

    // Define Halide Vars
    Func mm2("mm2"), tmp("tmp");
    Var m("m"), k ("k");
    RDom rm(0, nm, "nm");
    RDom rk(0, nk, "nk");
    RDom rj(0, nj, "nj");

    // Algorithm
    // out = (A * B) * (C * D)
    tmp1(m, k) += A(m, rk.x) * B(rk.x, k);
    tmp2(m, k) += C(m, rm.x) * D(rm.x, k);
    mm3(m, k) += tmp1(m, rj.x) * tmp2(rj.x, k);

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
