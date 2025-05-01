import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, parallel, get_loop
from pydsl.type import UInt32, F32, Index, AnyOp
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile, CTarget, PolyCTarget
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S
import numpy as np
from timeit import timeit
import subprocess
import ctypes

MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)

def transform_seq(targ: AnyOp):
    fuse_into(match(targ, 'fuse_target4'),
    fuse_into(match(targ, 'fuse_target3'),
    fuse_into(match(targ, 'fuse_target2'),
    fuse_into(match(targ, 'fuse_target1'),
              match(targ, 'fuse_1')))))

    fuse(match(targ, 'fuse_4'), match(targ, 'fuse_3'), 2)

    parallel(get_loop(tile(match(targ, 'tile'), [32, 32, 32], 6), 1), False)


# This example requires specific pass arguments to mlir-opt, which is currently not supported. Hence auto_build is set to false for now.

def lu(v0: Index, arg1: MemrefF32) -> UInt32:
    a: UInt32 = 5

    """@tag("tile")"""
    for arg2 in arange(v0):

        """@tag("fuse_4")"""
        for arg3 in arange(arg2):

            """@tag("fuse_1")"""
            for arg4 in arange(arg3):
                arg1[arg2, arg3] =    \
                    arg1[arg2, arg3]  \
                    - (arg1[arg2, arg4] 
                    * arg1[arg4, arg3])
            
            """@tag("fuse_target1")"""
            v1 = arg1[arg3, arg3]

            """@tag("fuse_target2")"""
            v2 = arg1[arg2, arg3]

            """@tag("fuse_target3")"""
            v3 = v2 / v1

            """@tag("fuse_target4")"""
            arg1[arg2, arg3] = v3

        """@tag("fuse_3")"""
        for arg3 in arange(arg2, v0):
            for arg4 in arange(arg2):
                arg1[arg2, arg3] =    \
                    arg1[arg2, arg3]  \
                    - (arg1[arg2, arg4] 
                    * arg1[arg4, arg3])

    return a


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
            case "CTEST":
                c_test = True
            case "NOCTEST":
                c_test = False
            case "DUMP_ARRAY":
                output_array = True
        if datasets.get(arg) is not None:
            current_dataset = arg
    n = datasets.get(current_dataset, 120)
    MemrefF32 = MemRefFactory((n, n), F32)

    lu = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(lu)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "lu.c", "-DNAIVE_VERSION", "-o", "lu-naive.so", "-lm"])
        lib = ctypes.CDLL('./lu-naive.so')
        lu_c = lib.kernel_lu
        
        lu_c.argtypes = [
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float) # A
        ]
            


    proper_init = n <= 120
    a = np.zeros((n, n)).astype(np.float32)
    # init array
    for i in range(n):
        for j in range(i+1):
            a[i, j] = -(j % n) / n + 1
        for j in range(i+1, n):
            a[i, j] = 0.0
        a[i, i] = 1.0

    # make the matrix positive semmi-definite
    if proper_init:
        b = np.zeros((n, n)).astype(np.float32)
        for r in range(n):
            for s in range(n):
                b[r, s] = 0.0
        for t in range(n):
            for r in range(n):
                for s in range(n):
                    b[r, s] += a[r, t] * a[s, t]
        for r in range(n):
            for s in range(n):
                a[r, s] = b[r, s]
    a_copy = a.copy()

    perf = timeit(lambda: lu(n, a), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: A", end="", file=sys.stderr)
        for i in range(n):
            for j in range(n):
                if (i * n + j) % 20 == 0: print("", file=sys.stderr)
                print(f"{a[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: A", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: lu_c(n, a_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            for j in range(n):
                if a_copy[i, j] - a[i, j] != 0: 
                    max_difference = max(max_difference, abs(a_copy[i, j] - a[i, j]))
        print(perf)
        if max_difference < 0.0001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "lu-naive.so"])