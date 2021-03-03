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
#define M 20 // 60 390 1900 1800
#define N 30 // 80 410 2100 2200

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
    Buffer<DTYPE> p(m, "vector p");
    Buffer<DTYPE> r(n, "vector r");
    Buffer<DTYPE> q(n, "vector p");
    Buffer<DTYPE> s(m, "vector p");
    srand(time(NULL));

    init_buffer(A, 2);
    init_buffer(p, 1);
    init_buffer(r, 1);

    // Define Halide Vars
    Func f_q("q"), f_s("s");
    Var v_m("m"), v_n("n");
    RDom r_n(0, n, "r_n");
    RDom r_m(0, m, "r_m");

    // Algorithm
    f_q(k) += A(r_m.x, k) * p(r_m.x);
    f_s(k) += A(k, r_n.x) * r(r_n.x);

    // Using basic schedule...Can be optimized
    f_q.print_loop_nest();
    f_s.print_loop_nest();

    // Compile
    f_q.compile_jit();
    f_s.compile_jit();

    // Execute or generate binary
    f_q.realize(q);
    f_s.realize(s);
    print_buffer(q, 1);
    print_buffer(s, 1);
    return 0;
}
