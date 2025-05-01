import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, parallel, reorder, distribute, get_loop, int_attr, unroll, recursively
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
import subprocess
import ctypes

MemrefF323D = MemRefFactory((DYNAMIC, DYNAMIC, DYNAMIC), F32)
MemrefF322D = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC, ), F32)

def transform_seq(targ: AnyOp):
    dist_res = distribute(match(targ, "distribute"), 2)
    tile_res = tile(get_loop(dist_res, 1), [32, 32], 4)
    reorder(get_loop(tile_res, 2), get_loop(tile_res, 3))

# 
# m,n,data,corr

def doitgen(nr: Index, nq: Index, np: Index, A: MemrefF323D, C4: MemrefF322D, sum: MemrefF321D) -> None:
    zero: F32 = 0.0
    for r in arange(nr):
        for q in arange(nq):
            """@tag("distribute")"""
            for p in arange(np):
                """@int_attr("set", 0)"""
                sum[p] = zero
                for s in arange(np):
                    with recursively(lambda x: int_attr(x, "set", 1)):
                        sum[p] = sum[p] + A[r, q, s] * C4[s, p]
            for p in arange(np):
                A[r, q, p] = sum[p]

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (8, 10, 12), "SMALL_DATASET": (20, 25, 30), "MEDIUM_DATASET": (40, 50, 60), "LARGE_DATASET": (140, 150, 160), "EXTRALARGE_DATASET": (220, 250, 270)}
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
    
    n_q, n_r, n_p = datasets.get(current_dataset, (20, 25, 30))

    MemrefF323D = MemRefFactory((n_r, n_q, n_p), F32)
    MemrefF322D = MemRefFactory((n_p, n_p), F32)
    MemrefF321D = MemRefFactory((n_p, ), F32)

    doitgen = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(doitgen)
    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "doitgen.c", "-DNAIVE_VERSION", "-o", "doitgen-naive.so", "-lm"])
        lib = ctypes.CDLL('./doitgen-naive.so')
        doitgen_c = lib.kernel_doitgen
        
        doitgen_c.argtypes = [
        ctypes.c_int,  # nr
        ctypes.c_int,  # nq
        ctypes.c_int,  # np
        ctypes.POINTER(ctypes.c_float), # A
        ctypes.POINTER(ctypes.c_float), # C4
        ctypes.POINTER(ctypes.c_float), # sum
        ]

    
    a = np.zeros((n_r, n_q, n_p)).astype(np.float32)
    sum_var = np.zeros(n_p).astype(np.float32)
    c4 = np.zeros((n_p, n_p)).astype(np.float32)
    
    # init array
    for i in range(n_r):
        for j in range(n_q):
            for k in range(n_p):
                a[i,j,k] = ((i*j + k) % n_p) / n_p
    for i in range(n_p):
        for j in range(n_p):
            c4[i, j] = (i*j % n_p) / n_p
    a_copy = a.copy()
    sum_var_copy = sum_var.copy()
    c4_copy = c4.copy()


    perf = timeit(lambda: doitgen(n_r, n_q, n_p, a, c4, sum_var), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: a", end="", file=sys.stderr)
        for i in range(n_r):
            for j in range(n_q):
                for k in range(n_p):
                    if ((i*n_q*n_p+j*n_p+k) % 20) == 0: print("", file=sys.stderr)
                    print(f"{a[i, j, k]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: a", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        sum_var_ptr = sum_var_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c4_ptr = c4_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


        perf = timeit(lambda: doitgen_c(n_r, n_q, n_p, a_ptr, c4_ptr, sum_var_ptr), number=1)
        max_difference = 0.0
        for i in range(n_r):
            for j in range(n_q):
                for k in range(n_p):
                    max_difference = max(max_difference, abs(a_copy[i, j, k] - a[i, j, k]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "doitgen-naive.so"])
