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
MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)
import numpy as np
from timeit import timeit
import subprocess
import ctypes


def transform_seq(targ: AnyOp):
    fuse_res1 = fuse_into(match(targ, 'fuse_target_1'), match(targ, 'fuse_1'))
    fuse_res2 = fuse_into(match(targ, 'fuse_target_2'), fuse_res1)
    fuse_res3 = fuse_into(match(targ, 'fuse_target_3'), fuse_res2)
    fuse_res4 = fuse_into(match(targ, 'fuse_target_4'), fuse_res3)
    fuse_res5 = fuse_into(match(targ, 'fuse_target_5'), fuse_res4)
    fuse_res6 = fuse_into(match(targ, 'fuse_target_6'), fuse_res5)
    fuse_res7 = fuse_into(match(targ, 'fuse_target_7'), fuse_res6)
    fuse_res8 = fuse_into(match(targ, 'fuse_target_8'), fuse_res7)
    fuse_res9 = fuse_into(match(targ, 'fuse_target_9'), fuse_res8)
    fuse_res10 = fuse_into(match(targ, 'fuse_target_10'), fuse_res9)
    fuse_res11 = fuse_into(match(targ, 'fuse_target_11'), fuse_res10)
    fuse_into(match(targ, 'fuse_target_12'), fuse_res11)
    tile_res = tile(match(targ, 'tile'), [32, 32, 32], 6)
    unroll(get_loop(tile_res, 3), 8)
    unroll(get_loop(tile_res, 4), 8)
    parallel(get_loop(tile_res, 1), False)


def cholesky(n: Index, A: MemrefF32) -> None:
    a: F32 = 1.0
    b: F32 = 0.0
    """@tag("tile")"""
    for i in arange(n):
        for j in arange(i):
            """@tag("fuse_1")"""
            for k in arange(j):
                A[i, j] = A[i, j] - A[i, k] * A[j, k]
            """@tag("fuse_target_1")"""
            A_ij = A[i, j]
            """@tag("fuse_target_2")"""
            A_jj = A[j, j]
            """@tag("fuse_target_3")"""
            A_div = A_ij / A_jj
            """@tag("fuse_target_4")"""
            A[i, j] =  A_div 
        for k in arange(i):
            """@tag("fuse_target_8")"""
            A_ik1 = A[i, k]
            """@tag("fuse_target_9")"""
            A_ik2 = A_ik1 * A_ik1
            """@tag("fuse_target_10")"""
            A_ii = A[i, i]
            """@tag("fuse_target_11")"""
            A_diff = A_ii - A_ik2
            """@tag("fuse_target_12")"""
            A[i, i] = A_diff
        """@tag("fuse_target_5")"""
        A_ii = A[i, i]
        """@tag("fuse_target_6")"""
        A_sqrt = sqrt(A_ii)
        """@tag("fuse_target_7")"""
        A[i, i] = A_sqrt


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

    cholesky = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(cholesky)


    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "cholesky.c", "-DNAIVE_VERSION", "-o", "cholesky-naive.so", "-lm"])
        lib = ctypes.CDLL('./cholesky-naive.so')
        cholesky_c = lib.kernel_cholesky
        
        cholesky_c.argtypes = [
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # a
        ]
    
    a = np.zeros((n, n)).astype(np.float32)
    
    # init array
    for i in range(n):
        for j in range(i+1):
            a[i, j] = -(j % n) / n + 1
        for j in range(i+1, n):
            a[i, j] = 0.0
        a[i, i] = 1.0

    # make the matrix positive semmi-definite
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

    perf = timeit(lambda: cholesky(n, a), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: a", end="", file=sys.stderr)
        for i in range(n):    
            for j in range(n):
                    if ((i * n + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{a[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: a", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: cholesky_c(n, a_ptr), number=1)
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
        subprocess.run(["rm", "cholesky-naive.so"])