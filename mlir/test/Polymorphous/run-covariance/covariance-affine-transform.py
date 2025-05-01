import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import recursively, tag, fuse_into, match_tag as match, fuse, tile, parallel, reorder, distribute, get_loop, int_attr, unroll
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

MemrefF32NM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32MM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32M = MemRefFactory((DYNAMIC, ), F32)

def transform_seq(targ: AnyOp):
    dist1_res = distribute(match(targ, 'dist1'), 3)
    tile1_res = tile(get_loop(dist1_res, 1), [32, 32], 4)
    reorder(get_loop(tile1_res, 2), get_loop(tile1_res, 3))
    parallel(get_loop(tile1_res, 0), False)
    parallel(get_loop(dist1_res, 0), False)
    parallel(get_loop(dist1_res, 2), False)

    tile2_res = tile(match(targ, 'tile_parallel'), [32, 32], 4)
    parallel(get_loop(tile2_res, 0), False)

    dist3_res = distribute(match(targ, 'dist3'), 3)
    tile3_res = tile(get_loop(dist3_res, 1), [32, 32, 32], 6)
    reorder_res = reorder(get_loop(tile3_res, 4), get_loop(tile3_res, 5))
    unroll(get_loop(reorder_res, 1), 8)
    parallel(get_loop(tile3_res, 0), False)
    parallel(get_loop(dist3_res, 2), False)
    parallel(get_loop(dist3_res, 0), False)
    
    
def covariance(m: Index, n: Index, float_n: F32, data: MemrefF32NM, cov: MemrefF32MM, mean: MemrefF32M) -> None:
    a: F32 = 1.0
    b: F32 = 0.0
    """@tag("dist1")"""
    for j in arange(m):
        """@int_attr("set", 0)"""
        mean[j] = b
        for i in arange(n):
            with recursively(lambda x: int_attr(x, "set", 1)):
                mean[j] = mean[j] + data[i, j]
        with  recursively(lambda x: int_attr(x, "set", 2)):
            mean[j] = mean[j] / float_n
    """@tag("tile_parallel")"""
    for i in arange(n):
        for j in arange(m):
            data[i, j] = data[i, j] - mean[j]
    """@tag("dist3")"""
    for i in arange(m):
        for j in arange(i, m):
            """@int_attr("set", 0)"""
            cov[i, j] = b
            for k in arange(n):
                with  recursively(lambda x: int_attr(x, "set", 1)):
                    cov[i, j] = cov[i, j] + data[k, i] * data[k, j]
            with  recursively(lambda x: int_attr(x, "set", 2)):
                cov[i, j] = cov[i, j] / (float_n - a)
            with  recursively(lambda x: int_attr(x, "set", 2)):
                cov[j, i] = cov[i, j]


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
    covariance = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(covariance)
    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "covariance.c", "-DNAIVE_VERSION", "-o", "covariance-naive.so", "-lm"])
        lib = ctypes.CDLL('./covariance-naive.so')
        covariance_c = lib.kernel_covariance
        
        covariance_c.argtypes = [
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.c_float,# float_n
        ctypes.POINTER(ctypes.c_float), # data
        ctypes.POINTER(ctypes.c_float), # cov
        ctypes.POINTER(ctypes.c_float), # mean
        ]
    float_n = np.float32(float(n))
    
    data = np.zeros((n, m)).astype(np.float32)
    cov = np.zeros((m, m)).astype(np.float32)
    mean = np.zeros(m).astype(np.float32)
    
    # init array
    for i in range(n):
        for j in range(m):
            data[i, j] = i * j / m
    
    data_copy = data.copy()
    cov_copy = cov.copy()
    mean_copy = mean.copy()

    perf = timeit(lambda: covariance(m, n, float_n, data, cov, mean), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: cov", end="", file=sys.stderr)
        for i in range(m):    
            for j in range(m):
                    if ((i * n + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{cov[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: cov", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        data_ptr = data_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cov_ptr = cov_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        mean_ptr = mean_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: covariance_c(m, n, float_n, data_ptr, cov_ptr, mean_ptr), number=1)
        max_difference = 0.0
        for i in range(m):
            for j in range(m):
                max_difference = max(max_difference, abs(cov_copy[i, j] - cov[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "covariance-naive.so"])