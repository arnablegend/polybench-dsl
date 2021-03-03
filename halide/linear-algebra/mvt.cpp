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
#define N 40 // 120 400 2000 4000

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
    Buffer<DTYPE> x1(n, "vector x1");
    Buffer<DTYPE> x2(n, "vector x2");
    Buffer<DTYPE> y1(n, "vector y1");
    Buffer<DTYPE> y2(n, "vector y2");
    srand(time(NULL));

    init_buffer(A, 2);
    init_buffer(y1, 1);
    init_buffer(y2, 1);

    // Define Halide Vars
    Func f_x1("x1"), f_x2("x2");
    Var k ("k");
    RDom r(0, n, "r");

    // Algorithm
    // out = (A * B) * (C * D)
    f_x1(k) += A(r.x, k) * y1(r.x);
    f_x2(k) += A(k, r.x) * y2(r.x);

    // Using basic schedule...Can be optimized
    f_x1.print_loop_nest();
    f_x2.print_loop_nest();

    // Compile
    f_x1.compile_jit();
    f_x2.compile_jit();

    // Execute or generate binary
    f_x1.realize(x1);
    f_x2.realize(x2);
    print_buffer(x1, 1);
    print_buffer(x2, 1);
    return 0;
}
