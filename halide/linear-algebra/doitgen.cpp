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
#define NQ 8 // 20 40 140 220
#define NR 10 // 25 50 150 250
#define NP 12 // 30 60 160 270

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
    }else if(num_dims == 3){
        for(int k = buf.min(2); k < buf.extent(2); k++){
            for(int j = buf.min(1); j < buf.extent(1); j++){
                for(int i = buf.min(0); i < buf.extent(0); i++)
                    cout << buf(i, j, k) << " ";
                cout << endl;
            }
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
    }else if(num_dims == 3){
        for(int k = buf.min(2); k < buf.extent(2); k++)
            for(int j = buf.min(1); j < buf.extent(1); j++)
                for(int i = buf.min(0); i < buf.extent(0); i++)
                    buf(i, j, k) = (DTYPE)((rand() % 100) / 10.0 + 1);
    }
}

int main(int argc, char** argv){
    int np=NP, nq=NQ, nr=NR;
    Buffer<DTYPE> A(np, nq, nr, "matrix A");
    Buffer<DTYPE> C4(np, np, "matrix C4");

    srand(time(NULL));
    init_buffer(A);
    init_buffer(C4);

    // Define Halide Vars
    Func f_a("produce a");
    Var p("p"), q("q"), r("r");
    RDom r_p(0, np, "r_p");

    // Algorithm
    tmp(p, q, r) += A(r_p, q, r) * C4(r_p, p);
    f_a(p, q, r) = tmp(p, q, r);

    // Using basic schedule...Can be optimized
    tmp.compute_root();
    f_a.print_loop_nest();

    // Compile
    f_a.compile_jit();

    // Execute or generate binary
    f_a.realize(A);
    print_buffer(A);
    return 0;
}
