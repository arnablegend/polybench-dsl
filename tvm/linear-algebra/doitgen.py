import tvm
from tvm import te

P = 12
Q = 8
R = 10


p = te.reduce_axis((0, P), "p")

A = te.placeholder((R, Q, P), name="A")
C4 = te.placeholder((P, P), name="C4")

doitgen = te.compute((R, Q, P), lambda z, y, x: te.sum(A[z, y, p] * C4[x, p], axis=p), name="comp")

sch = te.create_schedule(doitgen.op)

print(tvm.lower(sch, [A, C4, doitgen], simple_mode=True))

target = tvm.target.Target("llvm")
func = tvm.build(sch, [A, C4, doitgen], target=target, name="doitgen")
