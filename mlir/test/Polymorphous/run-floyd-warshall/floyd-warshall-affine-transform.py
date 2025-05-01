import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, parallel, reorder, distribute, get_loop, int_attr, recursively
from pydsl.type import UInt32, F64, Index, AnyOp, SInt32, Bool
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

MemrefSInt32 = MemRefFactory((DYNAMIC, DYNAMIC), SInt32)

def transform_seq(targ: AnyOp):
    tiled = tile(match(targ, "tile"), [32, 32], 4)
    parallel(get_loop(tiled, 0), True)

# 
# m,n,data,corr

def floyd_warshall(n: Index, path: MemrefSInt32) -> None:
    for k in arange(n):
        """@tag("tile")"""
        for i in arange(n):
            for j in arange(n):
                path[i, j] = path[i,j] if path[i,j] < path[i,k] + path[k,j] else path[i,k] + path[k,j]

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": 60, "SMALL_DATASET": 180, "MEDIUM_DATASET": 500, "LARGE_DATASET": 2800, "EXTRALARGE_DATASET": 5600}
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
    
    n = datasets.get(current_dataset, 180)
    MemrefSInt32 = MemRefFactory((n, n), SInt32)

    floyd_warshall = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(floyd_warshall)
    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "floyd-warshall.c", "-DNAIVE_VERSION", "-o", "floyd-warshall-naive.so", "-lm"])
        lib = ctypes.CDLL('./floyd-warshall-naive.so')
        floyd_warshall_c = lib.kernel_floyd_warshall
        
        floyd_warshall_c.argtypes = [
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # path
        ]
    
    path = np.zeros((n, n)).astype(np.int32)
    
    # init array
    for i in range(n):
        for j in range(n):
            path[i, j] = (i*j) % 7 + 1
            if ((i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0):
                path[i, j] = 999
        
    path_copy = path.copy()

    perf = timeit(lambda: floyd_warshall(n, path), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: path", end="", file=sys.stderr)
        for i in range(n):    
            for j in range(n):
                    if ((i * n + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{path[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: path", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        path_ptr = path_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: floyd_warshall_c(n, path_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            for j in range(n):
                max_difference = max(max_difference, abs(path_copy[i, j] - path[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "floyd-warshall-naive.so"])