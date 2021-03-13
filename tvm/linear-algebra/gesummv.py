import tvm
from tvm import te

N = 30

alpha = 1.5
beta = 1.2

n = te.reduce_axis((0, N), "n")
m = te.reduce_axis((0, N), "m")

A = te.placeholder((N, N), name="A")
B = te.placeholder((N, N), name="B")
X = te.placeholder((N,), name="X")

tmp = te.compute((N,), lambda x: te.sum(A[n, x] * X[n], axis=n), name="tmp")
tmp_y = te.compute((N,), lambda x: te.sum(B[x, m] * X[m], axis=m), name="tmp_y")
gesummv = te.compute((N,), lambda x: tmp(x) + tmp_y(x), name="gesummv")

sch = te.create_schedule(gesummv.op)
sch[tmp].compute_root()
sch[tmp_y].compute_root()


print(tvm.lower(sch, [A, B, X, gesummv], simple_mode=True))

target = tvm.target.Target("llvm")
func = tvm.build(sch, [A, B, X, gesummv], target=target, name="gesummv")
