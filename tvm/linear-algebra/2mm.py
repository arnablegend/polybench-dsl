import tvm
from tvm import te

I = 16
J = 18
K = 22
L = 24

dtype = 'float32'
alpha = 1.5
beta = 1.2

j = te.reduce_axis((0, J), "j")
k = te.reduce_axis((0, K), "k")

A = te.placeholder((I, J), name="A")
B = te.placeholder((J, K), name="B")
C = te.placeholder((K, L), name="C")
D = te.placeholder((I, L), name="D")

tmp = te.compute((I, K), lambda x, y: te.sum(A[x, j] * B[j, y], axis=j), name="tmp")
alpha_tmp = te.compute((I, K), lambda x, y: alpha * tmp[x, y], name="alpha_tmp")
beta_tmp = te.compute((I, L), lambda x, y: beta * D[x, y], name="beta_tmp")
compute_abc = te.compute((I, L), lambda x, y: te.sum(alpha_tmp[x, k] * C[k, y], axis=k), name="compute_abc")
mm2 = te.compute((I, L), lambda x, y: compute_abc[x, y] + beta_tmp[x, y], name="mm2")

sch = te.create_schedule(mm2.op)
sch[compute_abc].compute_root()
sch[beta_tmp].compute_inline()
sch[alpha_tmp].compute_inline()
sch[tmp].compute_root()


print(tvm.lower(sch, [A, B, C, D, mm2], simple_mode=True))

target = tvm.target.Target("llvm")
func = tvm.build(sch, [A, B, C, D, mm2], target=target, name="mm2")
