import tvm
from tvm import te

I = 16
J = 18
K = 20
L = 22
M = 24

dtype = 'float32'

j = te.reduce_axis((0, J), "j")
k = te.reduce_axis((0, K), "k")
m = te.reduce_axis((0, M), "m")

A = te.placeholder((K, I), name="A")
B = te.placeholder((J, K), name="B")
C = te.placeholder((M, J), name="C")
D = te.placeholder((L, M), name="D")

compute_ab = te.compute((J, I), lambda y, x: te.sum(A[k, x] * B[y, k], axis=k), name="compute_ab")
compute_cd = te.compute((L, J), lambda y, x: te.sum(C[m, x] * D[y, m], axis=m), name="compute_cd")
mm = te.compute((L, I), lambda y, x: te.sum(compute_ab[j, x] * compute_cd[y, j], axis=j), name="mm")

sch = te.create_schedule(mm.op)
sch[compute_ab].compute_root()
sch[compute_cd].compute_root()
print(tvm.lower(sch, [A, B, C, D, mm], simple_mode=True))

target = tvm.target.Target("llvm")
func = tvm.build(sch, [A, B, C, D, mm], target=target, name="mm")
