import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, parallel, reorder, distribute, get_loop, int_attr, recursively, unroll
from pydsl.type import UInt32, F32, Index, AnyOp, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile, CTarget, PolyCTarget
from pydsl.ops import sqrt
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S
import numpy as np
from timeit import timeit
import math
import subprocess
import ctypes

MemrefF32NM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32MM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32M = MemRefFactory((DYNAMIC, ), F32)

def transform_seq(targ: AnyOp):
    distribute_result1 = distribute(match(targ, "distribute1"), 3)
    parallel(get_loop(distribute_result1, 0), False)
    parallel(get_loop(distribute_result1, 2), False)

    tile_result1 = tile(get_loop(distribute_result1, 1), [32, 32], 4)
    reorder_res1 = reorder(get_loop(tile_result1, 2), get_loop(tile_result1, 3))
    unroll(get_loop(reorder_res1, 1), 8)
    parallel(get_loop(tile_result1, 0), False)


    distribute_result2 = distribute(match(targ, "distribute2"), 3)
    parallel(get_loop(distribute_result2, 0), False)
    parallel(get_loop(distribute_result2, 2), False)
    tile_result2 = tile(get_loop(distribute_result2, 1), [32, 32], 4)
    reorder_res2 = reorder(get_loop(tile_result2, 2), get_loop(tile_result2, 3))
    unroll(get_loop(reorder_res2, 1), 8)
    parallel(get_loop(tile_result2, 0), False)

    distribute_result = distribute(match(targ, 'tile_and_distribute'), 4)
    parallel(get_loop(distribute_result, 0), False)
    parallel(get_loop(distribute_result, 1), False)
    parallel(get_loop(distribute_result, 3), False)
    tile_result = tile(get_loop(distribute_result, 2), [32, 32, 32], 6)
    reorder_res = reorder(get_loop(tile_result, 4), get_loop(tile_result, 5))
    unroll(get_loop(reorder_res, 1), 8)
    parallel(get_loop(tile_result, 0), False)

    parallel(match(targ, "parallel"), False)
    

# 
# m,n,data,corr
def correlation(m: Index, n: Index, float_n: F32, data: MemrefF32NM, corr: MemrefF32MM, mean: MemrefF32M, stddev: MemrefF32M) -> None:    
    eps: F32 = 0.1

    """@tag("distribute1")"""
    for j in arange(m):
        """@int_attr("set", 0)"""
        mean[j] = F32(0.0)
        for i in arange(n):
            with recursively(lambda x: int_attr(x, "set", 1)):
                mean[j] = mean[j] + data[i, j]
        with  recursively(lambda x: int_attr(x, "set", 2)):
            mean[j] = mean[j] / float_n

    """@tag("distribute2")"""
    for j in arange(m):
        """@int_attr("set", 0)"""
        stddev[j] = F32(0.0)
        for i in arange(n):
            with  recursively(lambda x: int_attr(x, "set", 1)):
                stddev[j] = stddev[j] + (data[i, j] - mean[j]) * (data[i, j] - mean[j])
        with  recursively(lambda x: int_attr(x, "set", 2)):
            stddev[j] = stddev[j] / float_n
        with  recursively(lambda x: int_attr(x, "set", 2)):
            stddev[j] = sqrt(stddev[j])
        with  recursively(lambda x: int_attr(x, "set", 2)):
            stddev[j] = F32(1.0) if stddev[j] <= eps else stddev[j]

    """@tag("parallel")"""
    for i in arange(n):
        for j in arange(m):
            data[i, j] = data[i, j] - mean[j]
            data[i, j] = data[i, j] / (sqrt(float_n) * stddev[j])

    """@tag("tile_and_distribute")"""
    for i in arange(m - 1):
        """@int_attr("set", 0)"""
        corr[i, i] = F32(1.0)
        for j in arange(i+1, m):
            """@int_attr("set", 1)"""
            corr[i, j] = F32(0.0)
            for k in arange(n):
                with recursively(lambda x: int_attr(x, "set", 2)):
                    corr[i, j] = corr[i, j] + (data[k, i] * data[k, j])
            with recursively(lambda x: int_attr(x, "set", 3)):
                corr[j, i] = corr[i, j]
    
    corr[am(m-1, m-1)] = F32(1.0)


if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (28, 32), "SMALL_DATASET": (80, 100), "MEDIUM_DATASET": (240, 260), "LARGE_DATASET": (1200, 1400), "EXTRALARGE_DATASET": (2600, 3000)}
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
    
    m, n = datasets.get(current_dataset, (80, 100))
    MemrefF32NM = MemRefFactory((n, m), F32)
    MemrefF32MM = MemRefFactory((m, m), F32)
    MemrefF32M = MemRefFactory((m, ), F32)

    correlation = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(correlation)
    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "correlation.c", "-DNAIVE_VERSION", "-o", "correlation-naive.so", "-lm"])
        lib = ctypes.CDLL('./correlation-naive.so')
        correlation_c = lib.kernel_correlation
        
        correlation_c.argtypes = [
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.c_float,# float_n
        ctypes.POINTER(ctypes.c_float), # data
        ctypes.POINTER(ctypes.c_float), # corr
        ctypes.POINTER(ctypes.c_float), # mean
        ctypes.POINTER(ctypes.c_float), # stddev
        ]
    
    float_n = float(n)
    data = np.zeros((n, m)).astype(np.float32)
    corr = np.zeros((m, m)).astype(np.float32)
    mean = np.zeros(m).astype(np.float32)
    stddev = np.zeros(m).astype(np.float32)
    # init array
    for i in range(n):
        for j in range(m):
            data[i][j] = ((i * j)/m + i)

    data_copy = data.copy()
    corr_copy = corr.copy()
    mean_copy = mean.copy()
    stddev_copy = stddev.copy()

    perf = timeit(lambda: correlation(m, n, float_n, data, corr, mean, stddev), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: corr", end="", file=sys.stderr)
        for i in range(m):
            for j in range(m):
                if (i * m + j) % 20 == 0: print("", file=sys.stderr)
                print(f"{corr[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: corr", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        data_ptr = data_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        corr_ptr = corr_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        mean_ptr = mean_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        stddev_ptr = stddev_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: correlation_c(m, n, float_n, data_ptr, corr_ptr, mean_ptr, stddev_ptr), number=1)
        max_difference = 0.0
        for i in range(m):
            for j in range(m):
                max_difference = max(max_difference, abs(corr_copy[i, j] - corr[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        
        subprocess.run(["rm", "correlation-naive.so"])