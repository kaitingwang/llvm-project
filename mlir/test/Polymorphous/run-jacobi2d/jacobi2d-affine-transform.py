import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, unroll, get_loop, int_attr, distribute, reorder, parallel
from pydsl.type import UInt32, F32, Index, AnyOp
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile, CTarget, PolyCTarget
from pydsl.affine import    \
    affine_range as arange, \
    affine_map as am,       \
    dimension as D,         \
    symbol as S
import numpy as np
from timeit import timeit
import ctypes
import subprocess

MemRefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)

def transform_seq(targ: AnyOp):
    parallel(match(targ, "parallel1"), False)
    parallel(match(targ, "parallel2"), False)


def jacobi(T: Index, N: Index, a: MemRefF32, b: MemRefF32) -> None:
    for _ in arange(T):
        """@tag("parallel1")"""
        for i in arange(1, N - 1):
            for j in arange(1, N - 1):
                const:F32 = 0.2
                b[i, j] = (a[i, j] +        \
                                     a[i, j - 1] +    \
                                     a[i, j + 1] +    \
                                     a[i - 1, j] +    \
                                     a[i + 1, j]) * const

        """@tag("parallel2")"""
        for i in arange(1, N - 1):
            for j in arange(1, N - 1):
                const: F32 = 0.2
                a[i, j] = (b[i, j] +        \
                                     b[i, j - 1] +    \
                                     b[i, j + 1] +    \
                                     b[i - 1, j] +    \
                                     b[i + 1, j]) * const
if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (20, 30), "SMALL_DATASET": (40, 90), "MEDIUM_DATASET": (100, 250), "LARGE_DATASET": (500, 1300), "EXTRALARGE_DATASET": (1000, 2800)}
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

    tsteps, n = datasets.get(current_dataset, (40, 90))

    MemRefF32 = MemRefFactory((n, n), F32)

    jacobi = compile(locals(), dump_mlir=False, transform_seq=transform_seq, auto_build=True, target_class=compilation_target, dataset=current_dataset)(jacobi)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "jacobi2d.c", "-DNAIVE_VERSION", "-o", "jacobi2d-naive.so", "-lm"])
        lib = ctypes.CDLL('./jacobi2d-naive.so')
        jacobi2d_c = lib.kernel_jacobi_2d
        
        jacobi2d_c.argtypes = [
        ctypes.c_int,  # tsteps
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # a
        ctypes.POINTER(ctypes.c_float), # b
        ]
    
    a = np.zeros((n, n)).astype(np.float32)
    b = np.zeros((n, n)).astype(np.float32)
    
    # init array
    for i in range(n):
        for j in range(n):
            a[i, j] = (i*(j+2) + 2) / n
            b[i, j] = (i*(j+3) + 3) / n
    
    a_copy = a.copy()
    b_copy = b.copy()

    perf = timeit(lambda: jacobi(tsteps, n, a, b), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: A", end="", file=sys.stderr)
        for i in range(n):    
            for j in range(n):
                    if ((i * n + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{a[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: A", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: jacobi2d_c(tsteps, n, a_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            for j in range(n):
                max_difference = max(max_difference, abs(a_copy[i, j] - a[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "jacobi2d-naive.so"])