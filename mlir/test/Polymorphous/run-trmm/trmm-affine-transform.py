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

MemrefF32MM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32MN = MemRefFactory((DYNAMIC, DYNAMIC), F32)

def transform_seq(targ: AnyOp):
    dist_res1 = distribute(match(targ, "tile"), 2)
    tile_res1 = tile(get_loop(dist_res1, 0), [32, 32, 32], 6)
    reorder_res = reorder(get_loop(tile_res1, 0), get_loop(tile_res1, 1))
    inner_reorder_res = reorder(get_loop(tile_res1, 4), get_loop(tile_res1, 5))
    parallel(get_loop(reorder_res, 0), False)
    unroll(get_loop(inner_reorder_res, 1), 8)
    
    tile_res2 = tile(get_loop(dist_res1, 1), [32, 32], 4)
    parallel(get_loop(tile_res2, 0), False)
    unroll(get_loop(tile_res2, 2), 8)


def trmm(m: Index, n: Index, alpha: F32, A: MemrefF32MM, B: MemrefF32MN) -> None:
    """@tag("tile")"""
    for i in arange(m):
        for j in arange(n):
            for k in arange(i+1, m):
                with recursively(lambda x: int_attr(x, "set", 0)):
                    B[i, j] = B[i, j] + B[k, j] * A[k, i]
            with recursively(lambda x: int_attr(x, "set", 1)):
                B[i, j] = alpha * B[i, j]

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
            case "CTEST":
                c_test = True
            case "NOCTEST":
                c_test = False
            case "DUMP_ARRAY":
                output_array = True
        if datasets.get(arg) is not None:
            current_dataset = arg
    m, n = datasets.get(current_dataset, (60, 80))

    MemrefF32MM = MemRefFactory((m, m), F32)
    MemrefF32MN = MemRefFactory((m, n), F32)

    trmm = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(trmm)
    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "trmm.c", "-DNAIVE_VERSION", "-o", "trmm-naive.so", "-lm"])
        lib = ctypes.CDLL('./trmm-naive.so')
        trmm_c = lib.kernel_trmm
        
        trmm_c.argtypes = [
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.c_float,# alpha
        ctypes.POINTER(ctypes.c_float), # a
        ctypes.POINTER(ctypes.c_float), # b
        ]

    a = np.zeros((m, m)).astype(np.float32)
    b = np.zeros((m, n)).astype(np.float32)
    alpha = 1.5
    
    # init array
    for i in range(m):
        for j in range(i):
            a[i, j] = ((i+j) % m) / m
        a[i, i] = 1
        for j in range(n):
            b[i, j] = ((n+(i-j)) % n) / n

    a_copy = a.copy()
    b_copy = b.copy()

    perf = timeit(lambda: trmm(m, n, alpha, a, b), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: B", end="", file=sys.stderr)
        for i in range(m):    
            for j in range(n):
                    if ((i * m + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{b[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: B", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: trmm_c(m, n, alpha, a_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(m):
            for j in range(n):
                max_difference = max(max_difference, abs(b_copy[i, j] - b[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "trmm-naive.so"])