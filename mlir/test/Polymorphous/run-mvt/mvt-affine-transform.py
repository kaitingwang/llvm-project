import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, parallel, reorder, distribute, get_loop, int_attr, unroll
from pydsl.type import UInt32, F64, Index, AnyOp, F32
from pydsl.memref import MemRefFactory, DYNAMIC
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

MemrefF322D = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC, ), F32)

def transform_seq(targ: AnyOp):
    tile_res1 = tile(match(targ, "tile1"), [32, 32], 4)
    tile_res2 = tile(match(targ, "tile2"), [32, 32], 4)
    reorder_res = reorder(get_loop(tile_res2, 2), get_loop(tile_res2, 3))
    parallel(get_loop(tile_res1, 0), False)
    parallel(get_loop(tile_res2, 0), False)
    unroll(get_loop(tile_res1, 2), 8)
    unroll(get_loop(reorder_res, 1), 8)


# m,n,data,corr

def mvt(n: Index, x1: MemrefF321D, x2: MemrefF321D, y1: MemrefF321D, y2: MemrefF321D, A: MemrefF322D) -> None:
    """@tag("tile1")"""
    for i in arange(n):
        for j in arange(n):
            x1[i] = x1[i] + A[i, j] * y1[j]
    """@tag("tile2")"""
    for i in arange(n):
        for j in arange(n):
            x2[i] = x2[i] + A[i, j] * y2[j]

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

    MemrefF322D = MemRefFactory((n, n), F32)
    MemrefF321D = MemRefFactory((n, ), F32)

    mvt = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(mvt)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "mvt.c", "-DNAIVE_VERSION", "-o", "mvt-naive.so", "-lm"])
        lib = ctypes.CDLL('./mvt-naive.so')
        mvt_c = lib.kernel_mvt
        
        mvt_c.argtypes = [
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # x1
        ctypes.POINTER(ctypes.c_float), # x2
        ctypes.POINTER(ctypes.c_float), # y1
        ctypes.POINTER(ctypes.c_float), # y2
        ctypes.POINTER(ctypes.c_float), # a
        ]
    
    a = np.zeros((n, n)).astype(np.float32)
    x1 = np.zeros(n).astype(np.float32)
    x2 = np.zeros(n).astype(np.float32)
    y1 = np.zeros(n).astype(np.float32)
    y2 = np.zeros(n).astype(np.float32)

    # init array
    for i in range(n):
        x1[i] = (i % n) / n
        x2[i] = ((i+1)%n) / n
        y1[i] = ((i+3)%n) / n
        y2[i] = ((i+4)%n) / n
        for j in range(n):
            a[i, j] = (i*j % n) / n
    
    a_copy = a.copy()
    x1_copy = x1.copy()
    x2_copy = x2.copy()
    y1_copy = y1.copy()
    y2_copy = y2.copy()

    perf = timeit(lambda: mvt(n, x1, x2, y1, y2, a), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: x1", end="", file=sys.stderr)
        for i in range(n):    
            if ((i) % 20) == 0: print("", file=sys.stderr)
            print(f"{x1[i]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: x1", file=sys.stderr)
        print("begin dump: x2", end="", file=sys.stderr)
        for i in range(n):    
            if ((i) % 20) == 0: print("", file=sys.stderr)
            print(f"{x2[i]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: x2", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x1_ptr = x1_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x2_ptr = x2_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y1_ptr = y1_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y2_ptr = y2_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: mvt_c(n, x1_ptr, x2_ptr, y1_ptr, y2_ptr, a_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(x1_copy[i] - x1[i]))
        for i in range(n):
            max_difference = max(max_difference, abs(x2_copy[i] - x2[i]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "mvt-naive.so"])