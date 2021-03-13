import tvm
from tvm import te

M = 38
N = 42

m = te.reduce_axis((0, M), "m")
n = te.reduce_axis((0, N), "n")

A = te.placeholder((N, M), name="A")
B = te.placeholder((N,), name="B")

tmp = te.compute((M,), lambda x: te.sum(A[n, x] * B[n], axis=n), name="tmp")
atax = te.compute((N,), lambda x: te.sum(A[x, m] * tmp[m], axis=m), name="atax")

sch = te.create_schedule(atax.op)
sch[tmp].compute_root()


print(tvm.lower(sch, [A, B, atax], simple_mode=True))

target = tvm.target.Target("llvm")
func = tvm.build(sch, [A, B, atax], target=target, name="atax")
