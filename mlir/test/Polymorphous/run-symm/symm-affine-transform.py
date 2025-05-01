import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, unroll, get_loop, int_attr, recursively, distribute, reorder, parallel, vectorize
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

def transform_seq(targ: AnyOp):
    parallel(match(targ, "distribute"))

    



def symm(n: Index, m: Index, alpha: F32, beta: F32, C: MemrefF32, A: MemrefF32, B: MemrefF32, temp_arr: MemrefF32) -> None:
    for i in arange(m):
        """@tag("distribute")"""
        for j in arange(n):
            """@int_attr("set", 0)"""
            temp_arr[i, j] = F32(0.0)
            for k in arange(i):
                with recursively(lambda x: int_attr(x, "set", 1)):
                    C[k, j] = C[k, j] + alpha * B[i, j] * A[i, k]
                with recursively(lambda x: int_attr(x, "set", 2)):
                    temp_arr[i, j] = temp_arr[i, j] + B[k, j] * A[i, k]
            with recursively(lambda x: int_attr(x, "set", 2)):
                C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp_arr[i, j]

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (20, 30), "SMALL_DATASET": (60, 80), "MEDIUM_DATASET": (200, 240), "LARGE_DATASET": (1000, 1200), "EXTRALARGE_DATASET": (2000, 2600)}
    current_dataset = "SMALL_DATASET"
    output_array = False
    for arg in sys.argv:
        match arg:
            case "CTARGET":
                compilation_target = CTarget
            case "POLYCTARGET":
                compilation_target = PolyCTarget
                print("Symm Poly Target cannot be supported due to missing alloca op support")
                exit()
            case "CTEST":
                c_test = True
            case "NOCTEST":
                c_test = False
            case "DUMP_ARRAY":
                output_array = True
        if datasets.get(arg) is not None:
            current_dataset = arg
    
    symm = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target)(symm)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "symm.c", "-DNAIVE_VERSION", "-o", "symm-naive.so", "-lm"])
        lib = ctypes.CDLL('./symm-naive.so')
        symm_c = lib.kernel_symm
        
        symm_c.argtypes = [
        ctypes.c_int,  # tsteps
        ctypes.c_int,  # n
        ctypes.c_float,# alpha
        ctypes.c_float,# beta
        ctypes.POINTER(ctypes.c_float), # c
        ctypes.POINTER(ctypes.c_float), # a
        ctypes.POINTER(ctypes.c_float), # b
        ]

    m, n = datasets.get(current_dataset, (60, 80))
    alpha = 1.5
    beta = 1.2
    a = np.zeros((m, m)).astype(np.float32)
    b = np.zeros((m, n)).astype(np.float32)
    c = np.zeros((m, n)).astype(np.float32)
    temp_arr = np.zeros((m, n)).astype(np.float32)

    
    
    # init array
    for i in range(m):
        for j in range(n):
            c[i, j] = ((i+j) % 100) / m
            b[i, j] = ((n+i-j) % 100) / m
    for i in range(m):
        for j in range(i+1):
            a[i, j] = ((i+j) % 100) / m
        for j in range(i+1, m):
            a[i, j] = -999

    c_copy = c.copy()
    a_copy = a.copy()
    b_copy = b.copy()

    perf = timeit(lambda: symm(n, m, alpha, beta, c, a, b, temp_arr), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: C", end="", file=sys.stderr)
        for i in range(m):    
            for j in range(n):
                    if ((i * m + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{c[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: C", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        c_ptr = c_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: symm_c(m, n, alpha, beta, c_ptr, a_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(m):
            for j in range(n):
                max_difference = max(max_difference, abs(c_copy[i, j] - c[i, j]))
        print(perf)
        if max_difference == 0.0:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "symm-naive.so"])