import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, unroll, get_loop, int_attr, distribute, reorder, parallel, recursively
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

MemrefF32_IJ = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_IK = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_KJ = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_JL = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_JM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32_ML = MemRefFactory((DYNAMIC, DYNAMIC), F32)
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

    dist_res2 = distribute(match(targ, "distribute3"), 2)
    tile_res2 = tile(get_loop(dist_res2, 1), [32, 32, 32], 6)
    reorder(get_loop(tile_res2, 4), get_loop(tile_res2, 5))
    parallel(get_loop(tile_res2, 0), False)

    

    


def Threemm(ni: Index, nj: Index, nk: Index, nl: Index, nm: Index, E: MemrefF32_IJ, A: MemrefF32_IK, B: MemrefF32_KJ, F: MemrefF32_JL, C: MemrefF32_JM, D_arr: MemrefF32_ML, G: MemrefF32_IL) -> F32:
    b: F32 = 0.0
    """@tag("distribute1")"""
    for i in arange(S(ni)):
        for j in arange(S(nj)):
            """@int_attr("set", 0)"""
            E[i, j] = b
            for k in arange(S(nk)):
                with recursively(lambda x: int_attr(x, "set", 1)):
                    E[i, j] = E[i, j] + A[i, k] * B[k, j]
    """@tag("distribute2")"""
    for i in arange(S(nj)):
        for j in arange(S(nl)):
            """@int_attr("set", 0)"""
            F[i, j] = b
            for k in arange(S(nm)):
                with recursively(lambda x: int_attr(x, "set", 1)):
                    F[i, j] = F[i, j] + C[i, k] * D_arr[k, j]
    """@tag("distribute3")"""
    for i in arange(S(ni)):
        for j in arange(S(nl)):
            """@int_attr("set", 0)"""
            G[i, j] = b
            for k in arange(S(nj)):
                with recursively(lambda x: int_attr(x, "set", 1)):
                    G[i, j] = G[i, j] + E[i, k] * F[k, j]

    return b

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (16, 18, 20, 22, 24), "SMALL_DATASET": (40, 50, 60, 70, 80), "MEDIUM_DATASET": (180, 190, 200, 210, 220), "LARGE_DATASET": (800, 900, 1000, 1100, 1200), "EXTRALARGE_DATASET": (1600, 1800, 2000, 2200, 2400)}
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
    ni, nj, nk, nl, nm = datasets.get(current_dataset, (40, 50, 60, 70, 80))


    MemrefF32_IJ = MemRefFactory((ni, nj), F32)
    MemrefF32_IK = MemRefFactory((ni, nk), F32)
    MemrefF32_KJ = MemRefFactory((nk, nj), F32)
    MemrefF32_JL = MemRefFactory((nj, nl), F32)
    MemrefF32_JM = MemRefFactory((nj, nm), F32)
    MemrefF32_ML = MemRefFactory((nm, nl), F32)
    MemrefF32_IL = MemRefFactory((ni, nl), F32)

    Threemm = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(Threemm)
    
    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "3mm.c", "-DNAIVE_VERSION", "-o", "3mm-naive.so", "-lm"])
        lib = ctypes.CDLL('./3mm-naive.so')
        Threemm_c = lib.kernel_3mm
        
        Threemm_c.argtypes = [
        ctypes.c_int,  # ni
        ctypes.c_int,  # nj
        ctypes.c_int,  # nk
        ctypes.c_int,  # nl
        ctypes.c_int,  # nm
        ctypes.POINTER(ctypes.c_float), # E
        ctypes.POINTER(ctypes.c_float), # A
        ctypes.POINTER(ctypes.c_float), # B
        ctypes.POINTER(ctypes.c_float), # F
        ctypes.POINTER(ctypes.c_float), # C
        ctypes.POINTER(ctypes.c_float), # D
        ctypes.POINTER(ctypes.c_float)  # G
        ]
    
    
    e = np.zeros((ni, nj)).astype(np.float32)
    a = np.zeros((ni, nk)).astype(np.float32)
    b = np.zeros((nk, nj)).astype(np.float32)
    f = np.zeros((nj, nl)).astype(np.float32)
    c = np.zeros((nj, nm)).astype(np.float32)
    d = np.zeros((nm, nl)).astype(np.float32)
    g = np.zeros((ni, nl)).astype(np.float32)
    # init array
    alpha = 1.5
    beta = 1.2
    for i in range(ni):
        for j in range(nk):
            a[i, j] = ((i*j+1) % ni) / (5 * ni)
    for i in range(nk):
        for j in range(nj):
            b[i, j] = ((i*(j+1)+2) % nj) / (5 * nj)
    for i in range(nj):
        for j in range(nm):
            c[i, j] = ((i*(j+3))% nl) / (5*nl)
    for i in range(nm):
        for j in range(nl):
            d[i, j] = ((i*(j+2)+2) % nk) / (5*nk) 
    e_copy = e.copy()
    a_copy = a.copy()
    b_copy = b.copy()
    f_copy = f.copy()
    c_copy = c.copy()
    d_copy = d.copy()
    g_copy = g.copy()

    perf = timeit(lambda: Threemm(ni, nj, nk, nl, nm, e, a, b, f, c, d, g), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: G", end="", file=sys.stderr)
        for i in range(ni):
            for j in range(nl):
                if (i * ni + j) % 20 == 0: print("", file=sys.stderr)
                print(f"{g[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: G", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_ptr = c_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        d_ptr = d_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        e_ptr = e_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        f_ptr = f_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        g_ptr = g_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: Threemm_c(ni, nj, nk, nl, nm, e_ptr, a_ptr, b_ptr, f_ptr, c_ptr, d_ptr, g_ptr), number=1)
        max_difference = 0.0
        for i in range(ni):
            for j in range(nl):
                max_difference = max(max_difference, abs(g_copy[i, j] - g[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "3mm-naive.so"])
