import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, unroll, get_loop, int_attr, distribute, reorder, parallel
from pydsl.type import UInt32, F64, Index, AnyOp, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.ops import sqrt
from pydsl.frontend import compile, CTarget, PolyCTarget
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S
import numpy as np
from timeit import timeit
import ctypes
import subprocess

MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC,), F32)

def transform_seq(targ: AnyOp):
    parallel(match(targ, "parallel"))

    


def ludcmp(w: MemrefF321D, n: Index, A: MemrefF32, b: MemrefF321D, x: MemrefF321D, y: MemrefF321D) -> None:
    for i in arange(n):
        for j in arange(i):
            w[0] = A[i, j]
            for k in arange(j):
                w[0] = w[0] - A[i, k] * A[k, j]
            A[i, j] = w[0] / A[j, j]
        """@tag("parallel")"""
        for j in arange(i, n):
            w[0] = A[i, j]
            for k in arange(i):
                w[0] = w[0] - A[i, k] * A[k, j]
            A[i, j] = w[0]
    for i in arange(n):
        w[0] = b[i]
        for j in arange(i):
            w[0] = w[0] - A[i, j] * y[j]
        y[i] = w[0]
    
    for i in arange(n):
        w[0] = y[n-i-1]
        for j in arange(n-i, n):
            w[0] = w[0] - A[n-i-1, j] * x[j]
        x[n-i-1] = w[0] / A[n-i-1, n-i-1]

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": 40, "SMALL_DATASET": 120, "MEDIUM_DATASET": 400, "LARGE_DATASET": 2000, "EXTRALARGE_DATASET": 4000}
    current_dataset = "SMALL_DATASET"
    output_array = False
    for arg in sys.argv:
        match arg:
            case "CTARGET":
                compilation_target = CTarget
            case "POLYCTARGET":
                compilation_target = PolyCTarget
                print("Ludcmp Poly Target cannot be supported due to missing alloca op support")
                exit()
            case "CTEST":
                c_test = True
            case "NOCTEST":
                c_test = False
            case "DUMP_ARRAY":
                output_array = True
        if datasets.get(arg) is not None:
            current_dataset = arg

    ludcmp = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target)(ludcmp)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "ludcmp.c", "-DNAIVE_VERSION", "-o", "ludcmp-naive.so", "-lm"])
        lib = ctypes.CDLL('./ludcmp-naive.so')
        ludcmp_c = lib.kernel_ludcmp
        
        ludcmp_c.argtypes = [
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # a
        ctypes.POINTER(ctypes.c_float), # b
        ctypes.POINTER(ctypes.c_float), # x
        ctypes.POINTER(ctypes.c_float), # y
        ]
    
    n = datasets.get(current_dataset, 120)

    proper_init = n <= 120
    
    a = np.zeros((n, n)).astype(np.float32)
    b = np.zeros(n).astype(np.float32)
    x = np.zeros(n).astype(np.float32)
    y = np.zeros(n).astype(np.float32)
    w = np.zeros(1).astype(np.float32)

    a_copy = a.copy()
    b_copy = b.copy()
    x_copy = x.copy()
    y_copy = y.copy()
    
    # init array
    for i in range(n):
        b[i] = (i+1)/n/2 + 4
    for i in range(n):
        for j in range(i+1):
            a[i, j] = (-j % n) / n
            if (-j % n) / n == 0: # why is this how it works? This gives the same results as ludcmp.c for the values of a, but it shouldn't.
                a[i, j] = 1
        for j in range(i+1, n):
            a[i, j] = 0
        a[i,i] = 1

    # make the matrix positive semmi-definite
    if proper_init:
        b_temp = np.zeros((n, n)).astype(np.float32)
        for t in range(n):
            for r in range(n):
                for s in range(n):
                    b_temp[r, s] += a[r, t] * a[s, t]
        for r in range(n):
            for s in range(n):
                a[r, s] = b_temp[r, s]

    perf = timeit(lambda: ludcmp(w, n, a, b, x, y), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: x", end="", file=sys.stderr)
        for i in range(n): 
            if (i) %20 == 0: print("", file=sys.stderr)
            print(f"{x[i]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: x", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: ludcmp_c(n, a_ptr, b_ptr, x_ptr, y_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(x_copy[i] - x[i]))
        print(perf)
        if max_difference == 0.0:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "ludcmp-naive.so"])