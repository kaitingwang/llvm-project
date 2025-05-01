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

MemrefF32_IJ = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_IK = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_KJ = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_JL = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_IL = MemRefFactory((DYNAMIC, DYNAMIC), F32)

def transform_seq(targ: AnyOp):
    dist_res1 = distribute(match(targ, "distribute1"), 2)
    tile_res1 = tile(get_loop(dist_res1, 1), [32, 32, 32], 6)
    reorder(get_loop(tile_res1, 4), get_loop(tile_res1, 5))
    parallel(get_loop(tile_res1, 0), False)
    dist_res2 = distribute(match(targ, "distribute2"), 2)
    tile_res2 = tile(get_loop(dist_res2, 1), [32, 32, 32], 6)
    reorder(get_loop(tile_res2, 4), get_loop(tile_res2, 5))
    parallel(get_loop(tile_res2, 0), False)


def Twomm(ni: Index, nj: Index, nk: Index, nl: Index, alpha: F32, beta: F32, tmp: MemrefF32_IJ, A: MemrefF32_IK, B: MemrefF32_KJ, C: MemrefF32_JL, D_arr: MemrefF32_IL) -> None:
    b: F32 = 0.0
    """@tag("distribute1")"""
    for i in arange(S(ni)):
        for j in arange(S(nj)):
            """@int_attr("set", 0)"""
            tmp[i, j] = b
            for k in arange(nk):
                with recursively(lambda x: int_attr(x, "set", 1)):
                    tmp[i, j] = tmp[i, j] + alpha * A[i, k] * B[k, j]
    """@tag("distribute2")"""
    for i in arange(S(ni)):
        for j in arange(S(nl)):
            with recursively(lambda x: int_attr(x, "set", 0)):
                D_arr[i, j] = D_arr[i, j] * beta
            for k in arange(S(nj)):
                with recursively(lambda x: int_attr(x, "set", 1)):
                    D_arr[i, j] = D_arr[i, j] + tmp[i, k] * C[k, j]


if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (16, 18, 22, 24), "SMALL_DATASET": (40, 50, 70, 80), "MEDIUM_DATASET": (180, 190, 210, 220), "LARGE_DATASET": (800, 900, 1100, 1200), "EXTRALARGE_DATASET": (1600, 1800, 2200, 2400)}
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
    ni, nj, nk, nl = datasets.get(current_dataset, (40, 50, 70, 80))
    MemrefF32_IJ = MemRefFactory((ni, nj), F32)
    MemrefF32_IK = MemRefFactory((ni, nk), F32)
    MemrefF32_KJ = MemRefFactory((nk, nj), F32)
    MemrefF32_JL = MemRefFactory((nj, nl), F32)
    MemrefF32_IL = MemRefFactory((ni, nl), F32)
    
    Twomm = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(Twomm)
    
    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "2mm.c", "-DNAIVE_VERSION", "-o", "2mm-naive.so", "-lm"])
        lib = ctypes.CDLL('./2mm-naive.so')
        Twomm_c = lib.kernel_2mm
        
        Twomm_c.argtypes = [
        ctypes.c_int,  # ni
        ctypes.c_int,  # nj
        ctypes.c_int,  # nk
        ctypes.c_int,  # nl
        ctypes.c_float, # alpha
        ctypes.c_float, # beta
        ctypes.POINTER(ctypes.c_float), # tmp
        ctypes.POINTER(ctypes.c_float), # A
        ctypes.POINTER(ctypes.c_float), # B
        ctypes.POINTER(ctypes.c_float), # C
        ctypes.POINTER(ctypes.c_float)  # D
        ]


    tmp = np.zeros((ni, nj)).astype(np.float32)
    a = np.zeros((ni, nk)).astype(np.float32)
    b = np.zeros((nk, nj)).astype(np.float32)
    c = np.zeros((nj, nl)).astype(np.float32)
    d = np.zeros((ni, nl)).astype(np.float32)
    # init array
    alpha = 1.5
    beta = 1.2
    for i in range(ni):
        for j in range(nk):
            a[i, j] = ((i*j+1) % ni) / ni
    for i in range(nk):
        for j in range(nj):
            b[i, j] = (i*(j+1) % nj) / nj
    for i in range(nj):
        for j in range(nl):
            c[i, j] = ((i*(j+3)+1)% nl) / nl
    for i in range(ni):
        for j in range(nl):
            d[i, j] = (i*(j+2) % nk) / nk 
    
    a_copy = a.copy()
    b_copy = b.copy()
    c_copy = c.copy()
    d_copy = d.copy()

    perf = timeit(lambda: Twomm(ni, nj, nk, nl, alpha, beta, tmp, a, b, c, d), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: D", end="", file=sys.stderr)
        for i in range(ni):
            for j in range(nl):
                if (i * ni + j) % 20 == 0: print("", file=sys.stderr)
                print(f"{d[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: D", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_ptr = c_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        d_ptr = d_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        tmp_ptr = tmp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: Twomm_c(ni, nj, nk, nl, alpha, beta, tmp_ptr, a_ptr, b_ptr, c_ptr, d_ptr), number=1)
        max_difference = 0.0
        for i in range(ni):
            for j in range(nl):
                max_difference = max(max_difference, abs(d_copy[i, j] - d[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "2mm-naive.so"])