import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import recursively, tag, fuse_into, match_tag as match, fuse, tile, unroll, get_loop, int_attr, distribute, reorder, parallel
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
import subprocess
import ctypes

MemrefF32NM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32N = MemRefFactory((DYNAMIC,), F32)
MemrefF32M = MemRefFactory((DYNAMIC,), F32)

def transform_seq(targ: AnyOp):
    dist_res = distribute(match(targ, "distribute"), 3)
    tile_res = tile(get_loop(dist_res, 1), [32, 32], 4)
    tile2_res = tile(get_loop(dist_res, 2), [32, 32], 4)
    reorder_res = reorder(get_loop(tile2_res, 0), get_loop(tile_res, 1))
    parallel(get_loop(tile_res, 0), False)
    parallel(get_loop(reorder_res, 1), False)

    unroll(get_loop(tile_res, 2), 2)
    unroll(get_loop(tile2_res, 2), 2)


def atax(m: Index, n: Index, A: MemrefF32NM, x: MemrefF32N, y: MemrefF32N, tmp: MemrefF32M) -> None:
    b: F32 = 0.0
    for i in arange(n):
        y[i] = b
    """@tag("distribute")"""
    for i in arange(m):
        """@int_attr("set", 0)"""
        tmp[i] = b
        for j in arange(n):
            with recursively(lambda x: int_attr(x, "set", 1)):
                tmp[i] = tmp[i] + A[i, j] * x[j]
        for j in arange(n):
            with recursively(lambda x: int_attr(x, "set", 2)):
                y[j] = y[j] + A[i, j] * tmp[i]


if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (38, 42), "SMALL_DATASET": (116, 124), "MEDIUM_DATASET": (390, 410), "LARGE_DATASET": (1900, 2100), "EXTRALARGE_DATASET": (1800, 2200)}
    current_dataset = "SMALL_DATASET"
    output_array = False
    for arg in sys.argv:
        match arg:
            case "CTARGET":
                compilation_target = CTarget
            case "POLYCTARGET":
                compilation_target = PolyCTarget
                print("Atax Poly Target cannot be supported due to missing alloca op support")
                exit()
            case "CTEST":
                c_test = True
            case "NOCTEST":
                c_test = False
            case "DUMP_ARRAY":
                output_array = True
        if datasets.get(arg) is not None:
            current_dataset = arg

    m, n = datasets.get(current_dataset, (116, 124))
    MemrefF32NM = MemRefFactory((m, n), F32)
    MemrefF32N = MemRefFactory((n,), F32)
    MemrefF32M = MemRefFactory((m,), F32)

    atax = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(atax)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "atax.c", "-DNAIVE_VERSION", "-o", "atax-naive.so", "-lm"])
        lib = ctypes.CDLL('./atax-naive.so')
        atax_c = lib.kernel_atax
        
        atax_c.argtypes = [
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # A
        ctypes.POINTER(ctypes.c_float), # x
        ctypes.POINTER(ctypes.c_float), # y
        ctypes.POINTER(ctypes.c_float)  # tmp
        ]

    
    a = np.zeros((m, n)).astype(np.float32)
    x = np.zeros((n)).astype(np.float32)
    y = np.zeros((n)).astype(np.float32)
    tmp = np.zeros((m)).astype(np.float32)

    for i in range(n):
         x[i] = 1 + (i / n)
    for i in range(m):
         for j in range(n):
              a[i, j] = ((i+j) % n) / (5*m)
    
    a_copy = a.copy()
    x_copy = x.copy()
    y_copy = y.copy()
    tmp_copy = tmp.copy()

    perf = timeit(lambda: atax(m, n, a, x, y, tmp), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: U", end="", file=sys.stderr)
        for i in range(n):
                if (i % 20) == 0: print("", file=sys.stderr)
                print(f"{y[i]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: U", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        tmp_ptr = tmp_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: atax_c(m, n, a_ptr, x_ptr, y_ptr, tmp_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(y_copy[i] - y[i]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "atax-naive.so"])