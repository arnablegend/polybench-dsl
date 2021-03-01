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
    int n=N, m=M;
    Buffer<DTYPE> A(n, n, "matrix A");
    Buffer<DTYPE> u1(n, "vector u1");
    Buffer<DTYPE> u2(n, "vector u2");
    Buffer<DTYPE> v1(n, "vector v1");
    Buffer<DTYPE> v2(n, "vector v2");
    Buffer<DTYPE> y(n, "buffer y");
    Buffer<DTYPE> z(n, "buffer z");
    Buffer<DTYPE> x(n, "buffer x");
    Buffer<DTYPE> w(n, "buffer w");
    srand(time(NULL));

    init_buffer(A, 2);
    init_buffer(u1, 1);
    init_buffer(u2, 1);
    init_buffer(v1, 1);
    init_buffer(v2, 1);
    init_buffer(y, 1);
    init_buffer(z, 1);

    DTYPE alpha = 1.5;
    DTYPE beta = 1.2;

    // Define Halide Vars
    Func f_a("f_a"), f_x("x"), f_w("f_w");
    Var m("m"), k ("k");
    RDom r(0, n, "r");

    // Algorithm
    f_a(i, j) += u1(i) * v1(j);
    f_a(i, j) += u2(i) * v2(j);
    f_x(i) += beta * A(i, r.x) * y(r.x);
    f_x(i) += z(i);
    f_w(i) += alpha * A(r.x, i) * z(r.x);

    // Using basic schedule...Can be optimized

    // Compile
    f_a.compile_jit();
    f_x.compile_jit();
    f_w.compile_jit();

    // Execute or generate binary
    f_a.realize(A);
    f_x.realize(x);
    f_w.realize(w);
    print_buffer(A, 2);
    print_buffer(x, 1);
    print_buffer(w, 1);
    return 0;
}
