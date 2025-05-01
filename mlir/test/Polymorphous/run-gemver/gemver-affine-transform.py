import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, unroll, get_loop, int_attr, distribute, reorder, parallel, vectorize
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

MemrefF322D = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC,), F32)

def transform_seq(targ: AnyOp):
    tile1res = tile(match(targ, "tile1"), [32, 32], 4)
    tile2res = tile(match(targ, "tile2"), [32, 32], 4)
    reordered = reorder(get_loop(tile2res, 2), get_loop(tile2res, 3))
    tile3res = tile(match(targ, "tile3"), [32, 32], 4)
    parallel(get_loop(tile1res, 0), False)
    parallel(get_loop(tile2res, 0), False)
    parallel(get_loop(tile3res, 0), False)
    # The below gives a small error in result when using clang
    # unroll(get_loop(tile1res, 2), 8)
    # unroll(get_loop(reordered, 1), 8)
    # unroll(get_loop(tile3res, 2), 8)


def gemver(n: Index, alpha: F32, beta: F32, A: MemrefF322D, u1: MemrefF321D, v1: MemrefF321D, u2: MemrefF321D, v2: MemrefF321D, w: MemrefF321D, x: MemrefF321D, y: MemrefF321D, z: MemrefF321D) -> None:
    b: F32 = 0.0
    """@tag("tile1")"""
    for i in arange(n):
        for j in arange(n):
            A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]
    
    """@tag("tile2")"""
    for i in arange(n):
        for j in arange(n):
            x[i] = x[i] + beta * A[j, i] * y[j]
    
    for i in arange(n):
        x[i] = x[i] + z[i]

    """@tag("tile3")"""
    for i in arange(n):
        for j in arange(n):
            w[i] = w[i] + alpha * A[i, j] * x[j]

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
    MemrefF321D = MemRefFactory((n,), F32)

    gemver = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(gemver)

    if c_test:
        subprocess.run(["clang", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "gemver.c", "-DNAIVE_VERSION", "-o", "gemver-naive.so", "-lm", "-fPIC"])
        lib = ctypes.CDLL('./gemver-naive.so')
        gemver_c = lib.kernel_gemver
        
        gemver_c.argtypes = [
        ctypes.c_int,  # n
        ctypes.c_float,# alpha
        ctypes.c_float,# beta
        ctypes.POINTER(ctypes.c_float), # A
        ctypes.POINTER(ctypes.c_float), # u1
        ctypes.POINTER(ctypes.c_float), # v1
        ctypes.POINTER(ctypes.c_float), # u2
        ctypes.POINTER(ctypes.c_float), # v2
        ctypes.POINTER(ctypes.c_float), # w
        ctypes.POINTER(ctypes.c_float), # x
        ctypes.POINTER(ctypes.c_float), # y
        ctypes.POINTER(ctypes.c_float), # z
        ]
    
    a = np.zeros((n, n)).astype(np.float32)
    u1 = np.zeros(n).astype(np.float32)
    v1 = np.zeros(n).astype(np.float32)
    u2 = np.zeros(n).astype(np.float32)
    v2 = np.zeros(n).astype(np.float32)
    w = np.zeros(n).astype(np.float32)
    x = np.zeros(n).astype(np.float32)
    y = np.zeros(n).astype(np.float32)
    z = np.zeros(n).astype(np.float32)
    alpha = 1.5
    beta = 1.2
    
    # init array
    for i in range(n):
        u1[i] = i
        u2[i] = ((i+1)/n)/2
        v1[i] = ((i+1)/n)/4
        v2[i] = ((i+1)/n)/6
        y[i] = ((i+1)/n)/8
        z[i] = ((i+1)/n)/9
        x[i] = 0
        w[i] = 0
        for j in range(n):
            a[i, j] = (i*j % n) / n

    a_copy = a.copy()
    u1_copy = u1.copy()
    v1_copy = v1.copy()
    u2_copy = u2.copy()
    v2_copy = v2.copy()
    w_copy = w.copy()
    x_copy = x.copy()
    y_copy = y.copy()
    z_copy = z.copy()

    perf = timeit(lambda: gemver(n, alpha, beta, a, u1, v1, u2, v2, w, x, y, z), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: w", end="", file=sys.stderr)
        for i in range(n):    
            if ((i) % 20) == 0: print("", file=sys.stderr)
            print(f"{w[i]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: w", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        u1_ptr = u1_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v1_ptr = v1_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        u2_ptr = u2_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        v2_ptr = v2_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        w_ptr = w_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        z_ptr = z_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: gemver_c(n, alpha, beta, a_ptr, u1_ptr, v1_ptr, u2_ptr, v2_ptr, w_ptr, x_ptr, y_ptr, z_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(w_copy[i] - w[i]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "gemver-naive.so"])