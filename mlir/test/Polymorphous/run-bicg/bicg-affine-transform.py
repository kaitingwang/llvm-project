import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

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
import subprocess
import ctypes

MemrefF32NM = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32N = MemRefFactory((DYNAMIC,), F32)
MemrefF32M = MemRefFactory((DYNAMIC,), F32)

def transform_seq(targ: AnyOp):
    distribute_result = distribute(match(targ, 'tile_and_distribute'), 3)
    tile_result_1 = tile(get_loop(distribute_result, 1), [32, 32], 4)
    tile_result_2 = tile(get_loop(distribute_result, 2), [32, 32], 4)
    reorder_res = reorder(get_loop(tile_result_1, 0), get_loop(tile_result_1, 1))

    parallel(get_loop(reorder_res, 1), False)
    parallel(get_loop(tile_result_2, 0), False)

    unroll(get_loop(tile_result_1, 2), 8)
    unroll(get_loop(tile_result_2, 2), 8)
    


def bicg(m: Index, n: Index, A: MemrefF32NM, s: MemrefF32M, q: MemrefF32N, p: MemrefF32M, r: MemrefF32N) -> None:
    a: F32 = 1.0
    b: F32 = 0.0
    for i in arange(m):
        s[i] = b
    """@tag("tile_and_distribute")"""
    for i in arange(n):
        """@int_attr("set", 0)"""
        q[i] = b
        for j in arange(m):
            with recursively(lambda x: int_attr(x, "set", 1)):
                s[j] = s[j] + r[i] * A[i, j]
            
            with recursively(lambda x: int_attr(x, "set", 2)):
                q[i] = q[i] + p[j] * A[i, j]

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (38, 42), "SMALL_DATASET": (116, 124), "MEDIUM_DATASET": (390, 410), "LARGE_DATASET": (1900, 2100), "EXTRALARGE_DATASET": (1800, 2200)}
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

    m, n = datasets.get(current_dataset, (116, 124))
    MemrefF32NM = MemRefFactory((m, n), F32)
    MemrefF32N = MemRefFactory((n,), F32)
    MemrefF32M = MemRefFactory((m,), F32)
    
    bicg = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(bicg)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "bicg.c", "-DNAIVE_VERSION", "-o", "bicg-naive.so", "-lm"])
        lib = ctypes.CDLL('./bicg-naive.so')
        bicg_c = lib.kernel_bicg
        
        bicg_c.argtypes = [
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # A
        ctypes.POINTER(ctypes.c_float), # s
        ctypes.POINTER(ctypes.c_float), # q
        ctypes.POINTER(ctypes.c_float), # p
        ctypes.POINTER(ctypes.c_float), # r
        ]
    
    a = np.zeros((n, m)).astype(np.float32)
    s = np.zeros((m)).astype(np.float32)
    q = np.zeros((n)).astype(np.float32)
    p = np.zeros((m)).astype(np.float32)
    r = np.zeros((n)).astype(np.float32)
    
    for i in range(m):
        p[i] = (i % m) / m
    for i in range(n):
        r[i] = (i % n) / n
        for j in range(m):
            a[i, j] = (i * (j + 1) % n) / n

    a_copy = a.copy()
    s_copy = s.copy()
    q_copy = q.copy()
    p_copy = p.copy()
    r_copy = r.copy()

    perf = timeit(lambda: bicg(m, n, a, s, q, p, r), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: s", end="", file=sys.stderr)
        for i in range(m):
                if (i % 20) == 0: print("", file=sys.stderr)
                print(f"{s[i]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: s", file=sys.stderr)


        print("begin dump: q", end="", file=sys.stderr)
        for i in range(n):
                if (i % 20) == 0: print("", file=sys.stderr)
                print(f"{q[i]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: q", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        s_ptr = s_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        q_ptr = q_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        p_ptr = p_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        r_ptr = r_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: bicg_c(m, n, a_ptr, s_ptr, q_ptr, p_ptr, r_ptr), number=1)
        max_difference = 0.0
        for i in range(m):
                max_difference = max(max_difference, abs(s_copy[i] - s[i]))
        for i in range(n):
                max_difference = max(max_difference, abs(q_copy[i] - q[i]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "bicg-naive.so"])