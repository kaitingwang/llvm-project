import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import recursively, tag, fuse_into, match_tag as match, fuse, tile, unroll, get_loop, int_attr, distribute, reorder, parallel, vectorize
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

MemrefF32IJ = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32IK = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32KJ = MemRefFactory((DYNAMIC, DYNAMIC), F32)

def transform_seq(targ: AnyOp):
    dist_res1 = distribute(match(targ, "distribute"), 2)
    tile_res1 = tile(get_loop(dist_res1, 1), [32, 26, 40], 6)
    parallel(get_loop(tile_res1, 0), False)
    vectorize(match(targ, "vectorize"))


def gemm(ni: Index, nj: Index, nk: Index, alpha: F32, beta: F32, C: MemrefF32IJ, A: MemrefF32IK, B: MemrefF32KJ) -> None:
    b: F32 = 0.0
    """@tag("distribute")"""
    for i in arange(ni):
        for j in arange(nj):
            with recursively(lambda x: int_attr(x, "set", 0)):
                C[i, j] = C[i, j] * beta
        for k in arange(nk):
            """@tag("vectorize")"""
            for j in arange(nj):
                with recursively(lambda x: int_attr(x, "set", 1)):
                    C[i, j] = C[i, j] + alpha * A[i, k] * B[k, j]

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"ODD_DATASET": (1000, 1, 1200), "MINI_DATASET": (20, 25, 30), "SMALL_DATASET": (60, 70, 80), "MEDIUM_DATASET": (200, 220, 240), "LARGE_DATASET": (1000, 1100, 1200), "EXTRALARGE_DATASET": (2000, 2300, 2600)}
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
    
    ni, nj, nk = datasets.get(current_dataset, (60, 70, 80))

    MemrefF32IJ = MemRefFactory((ni, nj), F32)
    MemrefF32IK = MemRefFactory((ni, nk), F32)
    MemrefF32KJ = MemRefFactory((nk, nj), F32)
    
    gemm = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(gemm)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "gemm.c", "-DNAIVE_VERSION", "-o", "gemm-naive.so", "-lm"])
        lib = ctypes.CDLL('./gemm-naive.so')
        gemm_c = lib.kernel_gemm
        
        gemm_c.argtypes = [
        ctypes.c_int,  # ni
        ctypes.c_int,  # nj
        ctypes.c_int,  # nk
        ctypes.c_float,# alpha
        ctypes.c_float,# beta
        ctypes.POINTER(ctypes.c_float), # C
        ctypes.POINTER(ctypes.c_float), # A
        ctypes.POINTER(ctypes.c_float), # B
        ]
    
    a = np.zeros((ni, nk)).astype(np.float32)
    b = np.zeros((nk, nj)).astype(np.float32)
    c = np.zeros((ni, nj)).astype(np.float32)
    alpha = 1.5
    beta = 1.2
    # init array
    for i in range(ni):
        for j in range(nj):
            c[i, j] = ((i*j+1) % ni) / ni
    for i in range(ni):
        for j in range(nk):
            a[i, j] = (i*(j+1) % nk) / nk
    for i in range(nk):
        for j in range(nj):
            b[i, j] = (i*(j+2) % nj) / nj
    
    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()


    perf = timeit(lambda: gemm(ni, nj, nk, alpha, beta, c, a, b), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: C", end="", file=sys.stderr)
        for i in range(ni):    
            for j in range(nj):
                    if ((i * ni + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{c[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: C", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_ptr = c_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: gemm_c(ni, nj, nk, alpha, beta, c_ptr, a_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(ni):
            for j in range(nj):
                max_difference = max(max_difference, abs(c_copy[i, j] - c[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "gemm-naive.so"])