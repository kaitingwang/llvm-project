import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import parallel, tag, fuse_into, match_tag as match, fuse, tile, unroll, get_loop, int_attr
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
    parallel(match(targ, "parallel"), False)

    


def seidal(tsteps: Index, n: Index, A: MemrefF32) -> None:
    a: F32 = 9.0
    for t in arange(tsteps):
        """@tag("parallel")"""
        for i in arange(1, n - 1):
            for j in arange(1, n - 1):
                A[i, j] = (A[i-1, j-1] + A[i-1, j] + A[i-1, j+1] + \
                                     A[i, j-1] + A[i, j] + A[i, j+1] + \
                                     A[i+1, j-1] + A[i+1, j] + A[i+1, j+1])/a

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (20, 40), "SMALL_DATASET": (40, 120), "MEDIUM_DATASET": (100, 400), "LARGE_DATASET": (500, 2000), "EXTRALARGE_DATASET": (1000, 4000)}
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

    tsteps, n = datasets.get(current_dataset, (40, 120))

    MemrefF32 = MemRefFactory((n, n), F32)

    seidal = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(seidal)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "seidal.c", "-DNAIVE_VERSION", "-o", "seidal-naive.so", "-lm"])
        lib = ctypes.CDLL('./seidal-naive.so')
        seidal_c = lib.kernel_seidel_2d
        
        seidal_c.argtypes = [
        ctypes.c_int,  # tsteps
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # A
        ]
    
    a = np.zeros((n, n)).astype(np.float32)
    
    # init array
    for i in range(n):
         for j in range(n):
              a[i, j] = (i*(j+2)+2) / n
    
    a_copy = a.copy()

    perf = timeit(lambda: seidal(tsteps, n, a), number=1)
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

        perf = timeit(lambda: seidal_c(tsteps, n, a_ptr), number=1)
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
        subprocess.run(["rm", "seidal-naive.so"])