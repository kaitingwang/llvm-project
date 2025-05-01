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

MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC, DYNAMIC), F32)

def transform_seq(targ: AnyOp):
    parallel(match(targ, "parallel1"), False)
    parallel(match(targ, "parallel2"), False)
    

def heat(tsteps: Index, n: Index, A: MemrefF32, B: MemrefF32) -> F32:
    a: F32 = 2.0
    b: F32 = 0.125
    for t in arange(tsteps):
        """@tag("parallel1")"""
        for i in arange(1, n-1):
            for j in arange(1, n-1):
                for k in arange(1, n-1):
                    B[i,j,k] = b * (A[i+1,j,k] - a * A[i,j,k] + A[i-1,j,k]) + \
                                            b * (A[i,j+1,k] - a * A[i,j,k] + A[i,j-1,k]) + \
                                            b * (A[i,j,k+1] - a * A[i,j,k] + A[i,j,k-1]) + \
                                            A[i,j,k]
        """@tag("parallel2")"""
        for i in arange(1, n-1):
            for j in arange(1, n-1):
                for k in arange(1, n-1):
                    A[i,j,k] = b * (B[i+1,j,k] - a * B[i,j,k] + B[i-1,j,k]) + \
                                            b * (B[i,j+1,k] - a * B[i,j,k] + B[i,j-1,k]) + \
                                            b * (B[i,j,k+1] - a * B[i,j,k] + B[i,j,k-1]) + \
                                            B[i,j,k]

        

    return b

if __name__ == "__main__":
    compilation_target = CTarget
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (20, 10), "SMALL_DATASET": (40, 20), "MEDIUM_DATASET": (100, 40), "LARGE_DATASET": (500, 120), "EXTRALARGE_DATASET": (1000, 200)}
    current_dataset = "SMALL_DATASET"
    output_array = False
    for arg in sys.argv:
        match arg:
            case "CTARGET":
                compilation_target = CTarget
            case "POLYCTARGET":
                compilation_target = PolyCTarget
            case "CTEST":
                c_test = True
            case "NOCTEST":
                c_test = False
            case "DUMP_ARRAY":
                output_array = True
        if datasets.get(arg) is not None:
            current_dataset = arg

    tsteps, n = datasets.get(current_dataset, (40, 20))
    
    MemrefF32 = MemRefFactory((n, n, n), F32)
    heat = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(heat)
    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "heat.c", "-DNAIVE_VERSION", "-o", "heat-naive.so", "-lm"])
        lib = ctypes.CDLL('./heat-naive.so')
        heat_c = lib.kernel_heat_3d
        
        heat_c.argtypes = [
        ctypes.c_int,  # tsteps
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # A
        ctypes.POINTER(ctypes.c_float), # B
        ]

    A = np.fromfunction(lambda i, j, k: (i + j + (n-k))* 10 / (n), (n, n, n), dtype=np.float32)
    B = np.empty_like(A)
    B[:] = A
    A_copy = A.copy()
    B_copy = B.copy()
    perf = timeit(lambda: heat(tsteps, n, A, B), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: A", end="", file=sys.stderr)
        for i in range(n):    
            for j in range(n):
                for k in range(n):
                    if ((i * n * n + j * n + k) % 20 == 0): print("", file=sys.stderr)
                    print(f"{A[i, j, k]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: A", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = A_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = B_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: heat_c(tsteps, n, a_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    max_difference = max(max_difference, abs(A_copy[i, j, k] - A[i, j, k]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "heat-naive.so"])