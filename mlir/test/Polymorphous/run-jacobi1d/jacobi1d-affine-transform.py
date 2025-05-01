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

MemRefF32 = MemRefFactory((DYNAMIC,), F32)

def transform_seq(targ: AnyOp):
    fuse(match(targ, "fuse_1"), match(targ, "fuse_2"), 1)
    tiled = tile(match(targ, "tile"), [32, 32], 4)
    parallel(get_loop(tiled, 1), False)

def jacobi1d(T: Index, N: Index, A: MemRefF32, B: MemRefF32) -> None:
    cst1: F32 = 0.33333

    """@tag("tile")"""
    for t in arange(T):
        """@tag("fuse_1")"""
        for i in arange(1, N - 1):
            B[i] = cst1 * (A[i-1] + A[i] + A[i+1])
        """@tag("fuse_2")"""
        for i in arange(1, N - 1):
            A[i] = cst1 * (B[i-1] + B[i] + B[i+1])

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (20, 30), "SMALL_DATASET": (40, 120), "MEDIUM_DATASET": (100, 400), "LARGE_DATASET": (500, 2000), "EXTRALARGE_DATASET": (1000, 4000)}
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
    
    tsteps, n = datasets.get(current_dataset, (40,120))

    MemRefF32 = MemRefFactory((n,), F32)
    jacobi_1d = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(jacobi1d)
    
    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "jacobi1d.c", "-DNAIVE_VERSION", "-o", "jacobi1d-naive.so", "-lm"])
        lib = ctypes.CDLL('./jacobi1d-naive.so')
        jacobi1d_c = lib.kernel_jacobi_1d
        
        jacobi1d_c.argtypes = [
        ctypes.c_int,  # tsteps
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # a
        ctypes.POINTER(ctypes.c_float), # b
        ]
    
    a = np.zeros(n).astype(np.float32)
    b = np.zeros(n).astype(np.float32)
    
    # init array
    for i in range(n):
         a[i] = (i+2) / n
         b[i] = (i+3) / n

    a_copy = a.copy()
    b_copy = b.copy()
    
    perf = timeit(lambda: jacobi_1d(tsteps, n, a, b), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: A", end="", file=sys.stderr)
        for i in range(n):    
            if ((i) % 20) == 0: print("", file=sys.stderr)
            print(f"{a[i]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: A", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


        perf = timeit(lambda: jacobi1d_c(tsteps, n, a_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(a_copy[i] - a[i]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "jacobi1d-naive.so"])