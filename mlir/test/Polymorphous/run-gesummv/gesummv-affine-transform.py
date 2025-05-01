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
import ctypes
import subprocess

MemrefF322D = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC,), F32)

def transform_seq(targ: AnyOp):
    dist_res = distribute(match(targ, "distribute"), 4)
    parallel(get_loop(dist_res, 0), False)
    parallel(get_loop(dist_res, 1), False)
    parallel(get_loop(dist_res, 2), False)
    parallel(get_loop(dist_res, 3), False)


def gesummv(n: Index, alpha: F32, beta: F32, A: MemrefF322D, B: MemrefF322D, tmp: MemrefF321D, x: MemrefF321D, y: MemrefF321D) -> None:
    b: F32 = 0.0
    """@tag("distribute")"""
    for i in arange(n):
        """@int_attr("set", 0)"""
        tmp[i] = b
        """@int_attr("set", 0)"""
        y[i] = b
        for j in arange(n):
            with recursively(lambda x: int_attr(x, "set", 1)):
                tmp[i] = A[i, j] * x[j] + tmp[i]
            with recursively(lambda x: int_attr(x, "set", 2)):
                y[i] = B[i, j] * x[j] + y[i]
        with recursively(lambda x: int_attr(x, "set", 3)):
            y[i] = alpha * tmp[i] + beta * y[i]

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": 30, "SMALL_DATASET": 90, "MEDIUM_DATASET": 250, "LARGE_DATASET": 1300, "EXTRALARGE_DATASET": 2800}
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

    n = datasets.get(current_dataset, 90)
    MemrefF322D = MemRefFactory((n, n), F32)
    MemrefF321D = MemRefFactory((n, ), F32)
    gesummv = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(gesummv)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "gesummv.c", "-DNAIVE_VERSION", "-o", "gesummv-naive.so", "-lm"])
        lib = ctypes.CDLL('./gesummv-naive.so')
        gesummv_c = lib.kernel_gesummv
        
        gesummv_c.argtypes = [
        ctypes.c_int,  # n
        ctypes.c_float,# alpha
        ctypes.c_float,# beta
        ctypes.POINTER(ctypes.c_float), # a
        ctypes.POINTER(ctypes.c_float), # b
        ctypes.POINTER(ctypes.c_float), # tmp
        ctypes.POINTER(ctypes.c_float), # x
        ctypes.POINTER(ctypes.c_float), # y
        ]
    
    a = np.zeros((n, n)).astype(np.float32)
    b = np.zeros((n, n)).astype(np.float32)
    tmp = np.zeros(n).astype(np.float32)
    x = np.zeros(n).astype(np.float32)
    y = np.zeros(n).astype(np.float32)
    alpha = 1.5
    beta = 1.2
    
    # init array
    for i in range(n):
        x[i] = (i % n) / n
        for j in range(n):
            a[i, j] = ((i*j+1) % n) / n
            b[i, j] = ((i*j+2) % n) / n
    
    a_copy = a.copy()
    b_copy = b.copy()
    tmp_copy = tmp.copy()
    x_copy = x.copy()
    y_copy = y.copy()

    perf = timeit(lambda: gesummv(n, alpha, beta, a, b, tmp, x, y), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: y", end="", file=sys.stderr)
        for i in range(n):    
                if ((i) % 20) == 0: print("", file=sys.stderr)
                print(f"{y[i]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: y", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        tmp_ptr = tmp_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: gesummv_c(n, alpha, beta, a_ptr, b_ptr, tmp_ptr, x_ptr, y_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(y_copy[i] - y[i]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "gesummv-naive.so"])