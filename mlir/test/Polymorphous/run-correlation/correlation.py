from statistics import mean, stdev
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.ops import sqrt
from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, parallel, reorder, distribute, get_loop, int_attr, recursively, unroll
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
import math

MemrefNMF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefMMF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefMF32 = MemRefFactory((DYNAMIC,), F32)

M = 1200
N = 1400

def transform_seq(targ: AnyOp):
    distribute_result1 = distribute(match(targ, 'distribute_tile1'), 3)
    tile_result1 = tile(get_loop(distribute_result1, 1), [32, 32], 4)
    parallel(get_loop(tile_result1, 0), False)
    unroll(get_loop(tile_result1, 3), 8)
    
    distribute_result2 = distribute(match(targ, 'distribute_tile2'), 3)
    tile_result2 = tile(get_loop(distribute_result2, 1), [32, 32], 4)
    parallel(get_loop(tile_result2, 0), False)
    unroll(get_loop(tile_result2, 3), 8)

    distribute_result = distribute(match(targ, 'distribute_tile_reorder'), 4)
    tile_result = tile(get_loop(distribute_result, 2), [32, 32, 32], 6)
    reorder_res = reorder(get_loop(tile_result, 4), get_loop(tile_result, 5))
    parallel(get_loop(tile_result, 0), False)
    unroll(get_loop(reorder_res, 1), 8)
    

def correlation_python(m: Index, n: Index, float_n: F32, data: MemrefNMF32, corr: MemrefMMF32, mean: MemrefMF32, stddev: MemrefMF32) -> None:    
    a: F32 = 1.0
    b: F32 = 0.0
    eps: F32 = 0.1
    """@tag("distribute_tile1")"""
    for j in arange(m):
        """@int_attr("set", 0)"""
        mean[j] = b 
        for i in arange(n):
            with recursively(lambda x: int_attr(x, "set", 1)):
                mean[j] = mean[j] + data[i, j]
        with recursively(lambda x: int_attr(x, "set", 2)):
            mean[j] = mean[j] / float_n
    
    """@tag("distribute_tile2")"""
    for j in arange(m):
        """@int_attr("set", 0)"""
        stddev[j] = b 
        for i in arange(n):
            with recursively(lambda x: int_attr(x, "set", 1)):
                stddev[j] = stddev[j] + (data[i, j] - mean[j]) * (data[i, j] - mean[j])
        with recursively(lambda x: int_attr(x, "set", 2)):
            stddev[j] = stddev[j] / float_n
        with recursively(lambda x: int_attr(x, "set", 2)):
            stddev[j] = sqrt(stddev[j])
        with recursively(lambda x: int_attr(x, "set", 2)):
            stddev[j] = a if stddev[j] <= eps else stddev[j]

    for i in arange(n):
        for j in arange(m):
            data[i, j] = data[i, j] - mean[j]
            data[i, j] = data[i, j] / (sqrt(float_n) * stddev[j])

    """@tag("distribute_tile_reorder")"""
    for i in arange(m-1):
        """@int_attr("set", 0)"""
        corr[i, i] = a
        for j in arange(i+1, m):
            """@int_attr("set", 1)"""
            corr[i, j] = b
            for k in arange(n):
                with recursively(lambda x: int_attr(x, "set", 2)):
                    corr[i, j] = corr[i, j] + (data[k, i] * data[k, j])
            with recursively(lambda x: int_attr(x, "set", 3)):
                corr[j, i] = corr[i, j]
    corr[am(m-1, m-1)] = a

def floord(n,d):
    return -((-(n)+(d)-1)//(d)) if (n)<0 else (n)//(d)

def correlation_helper(m, n, float_n, data, mean, stddev):
    eps = 0.1
    for j in range(m):
        mean[j] = 0
        stddev[j] = 0.0
    lbp = 0
    ubp = floord(m-1, 32)
    t4_p = 0
    for t2 in range(lbp, ubp+1):
        for t3 in range(floord(n-1, 32)+1):
            for t4 in range(32*t3, min(n-1, 32*t3+32)-6, 8):
                lbv = 32*t2
                ubv = min(m-1, 32*t2+31)
                for t5 in range(lbv, ubv+1):
                    mean[t5] += data[t4, t5]
                    mean[t5] += data[t4+1, t5]
                    mean[t5] += data[t4+2, t5]
                    mean[t5] += data[t4+3, t5]
                    mean[t5] += data[t4+4, t5]
                    mean[t5] += data[t4+5, t5]
                    mean[t5] += data[t4+6, t5]
                    mean[t5] += data[t4+7, t5]
                t4_p = t4 + 8
            for t4 in range(t4_p, min(n-1, 32*t3+31)+1):
                lbv = 32*t2
                ubv = min(m-1, 32*t2+32)
                for t5 in range(lbv, ubv+1):
                    mean[t5] += data[t4, t5]
    for j in range(m):
        mean[j] /= float_n
    lbp = 0
    ubp = floord(m-1, 32)
    for t2 in range(lbp, ubp+1):
        for t3 in range(0, floord(n-1, 32)+1):
            for t4 in range(32*t3, min(n-1, 32*t3+31)-6, 8):
                lbv=32*t2
                ubv=min(m-1, 32*t2+31)
                for t5 in range(lbv, ubv+1):
                    stddev[t5] += (data[t4, t5] - mean[t5])   * (data[t4, t5] - mean[t5])
                    stddev[t5] += (data[t4+1, t5] - mean[t5]) * (data[(t4+1), t5] - mean[t5])
                    stddev[t5] += (data[t4+2, t5] - mean[t5]) * (data[(t4+2), t5] - mean[t5])
                    stddev[t5] += (data[t4+3, t5] - mean[t5]) * (data[(t4+3), t5] - mean[t5])
                    stddev[t5] += (data[t4+4, t5] - mean[t5]) * (data[(t4+4), t5] - mean[t5])
                    stddev[t5] += (data[t4+5, t5] - mean[t5]) * (data[(t4+5), t5] - mean[t5])
                    stddev[t5] += (data[t4+6, t5] - mean[t5]) * (data[(t4+6), t5] - mean[t5])
                    stddev[t5] += (data[t4+7, t5] - mean[t5]) * (data[(t4+7), t5] - mean[t5])
                t4_p = t4 + 8
            for t4 in range(t4_p, min(n-1, 32*t3+31)+1):
                lbv=32*t2
                ubv=min(m-1, 32*t2+31)
                for t5 in range(lbv, ubv+1):
                    stddev[t5] += (data[t4,t5] - mean[t5]) * (data[t4, t5] - mean[t5])
    lbp = 0
    ubp = floord(m-1, 32)
    for t2 in range(lbp, ubp+1):
        lbv = 32*t2
        ubv = min(m-1, 32*t2+31)
        for t3 in range(lbv, ubv+1):
            stddev[t3] /= float_n
            stddev[t3] = math.sqrt(stddev[t3])
            stddev[t3] = 1.0 if stddev[t3] <= eps else stddev[t3]
                

def kernel_correlation_local(m, n, float_n, data, corr, mean, stddev, func_correlation):
    eps = 0.1

    correlation_helper(m, n, float_n, data, mean, stddev)
    once = math.sqrt(float_n)
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= once * stddev[j]
    func_correlation(m, n, data, corr)


def init_array(n, m, data):
    for i in range(n):
        for j in range(m):
            data[i][j] = ((i * j)/m + i)


# ---------------------------------------------------- Running ----------------------------------------------------

def check_correctness(func, dump_file_name="python.log"):
    m = 80
    n = 100
    float_n = float(n)
    data = np.zeros((N, M)).astype(np.float32)
    corr = np.zeros((M, M)).astype(np.float32)
    mean = np.zeros(M).astype(np.float32)
    stddev = np.zeros(M).astype(np.float32)
    
    init_array(n, m, data)

    func(m, n, float_n, data, corr, mean, stddev)

    file = open(dump_file_name, "w")
    print("==BEGIN DUMP_ARRAYS==", file=file)
    print("begin dump: corr", end="", file=file)
    for i in range(m):
        for j in range(m):
            if (i * m + j) % 20 == 0: print("", file=file)
            print(f"{corr[i, j]:.2f} ", end="", file=file)
    print("\nend   dump: corr", file=file)
    print("==END   DUMP_ARRAYS==", file=file)


def measure_performance(func):
    m = M
    n = N
    float_n = float(n)
    data = np.zeros((N, M)).astype(np.float32)
    corr = np.zeros((M, M)).astype(np.float32)
    mean_array = np.zeros(M).astype(np.float32)
    stddev = np.zeros(M).astype(np.float32)
    
    init_array(n, m, data)

    perf = timeit(lambda: func(m, n, float_n, data, corr, mean_array, stddev), number=1)
    
    print(f"\n", file=sys.stdout)
    print(f"Correlation running with LARGE DATASET took {perf:.6f} seconds.", file=sys.stdout)
    print(f"\n", file=sys.stdout)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python correlation.py debug|perf")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "debug":
        M = 80
        N = 100
        MemrefNMF32 = MemRefFactory((N, M), F32)
        MemrefMMF32 = MemRefFactory((M, M), F32)
        MemrefMF32 = MemRefFactory((M,), F32)
        correlation_slow = compile(locals(), transform_seq=None, dump_mlir=False, auto_build=True, target_class=CTarget)(correlation_python)
        correlation_fast = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=PolyCTarget)(correlation_python)

        print(f"ensure algorithmic correctness\n")
        check_correctness(correlation_slow, "python_naive.log")
        check_correctness(correlation_fast, "python_opt.log")
    elif arg == "perf":
        M = 1200
        N = 1400
        MemrefNMF32 = MemRefFactory((N, M), F32)
        MemrefMMF32 = MemRefFactory((M, M), F32)
        MemrefMF32 = MemRefFactory((M,), F32)
        correlation_slow = compile(locals(), transform_seq=None, dump_mlir=False, auto_build=True, target_class=CTarget)(correlation_python)
        correlation_fast = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=PolyCTarget)(correlation_python)

        print(f"running correlation slow\n")
        measure_performance(correlation_slow)
        print(f"running correlation fast\n")
        measure_performance(correlation_fast)
    else:
        print("Usage: python correlation.py debug|perf")
        sys.exit(1)

