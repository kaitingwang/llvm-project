import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, unroll, get_loop, int_attr, recursively, distribute, reorder, parallel, vectorize
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
MemrefF321D = MemRefFactory((DYNAMIC, ), F32)

def transform_seq(targ: AnyOp):
    fuse_res1 = fuse_into(match(targ, "fuse_target1"), match(targ, "fuse_1"))
    fuse_res2 = fuse_into(match(targ, "fuse_target2"), fuse_res1)
    fuse_res3 = fuse_into(match(targ, "fuse_target3"), fuse_res2)
    fuse_into(match(targ, "fuse_target4"), fuse_res3)
    splited = distribute(match(targ, "distribute"), 2)
    tile_res = tile(get_loop(splited, 1), [32, 32], 4)
    unroll(get_loop(tile_res, 2), 8)
    parallel(get_loop(tile_res, 1), False)


def trisolv(n: Index, L: MemrefF322D, x: MemrefF321D, b: MemrefF321D) -> None:
    """@tag("distribute")"""
    for i in arange(n):
        with recursively(lambda x: int_attr(x, "set", 0)):
            x[i] = b[i]
        """@tag("fuse_1")"""
        for j in arange(i):
            with recursively(lambda x: int_attr(x, "set", 1)):
                x[i] = x[i] - L[i, j] * x[j]
        """@tag("fuse_target1")"""
        """@int_attr("set", 1)"""
        a = x[i]
        """@tag("fuse_target2")"""
        """@int_attr("set", 1)"""
        b = L[i, i]
        """@tag("fuse_target3")"""
        """@int_attr("set", 1)"""
        divf = a / b
        """@tag("fuse_target4")"""
        """@int_attr("set", 1)"""
        x[i] = divf

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
    MemrefF321D = MemRefFactory((n, ), F32)
    
    trisolv = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(trisolv)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "trisolv.c", "-DNAIVE_VERSION", "-o", "trisolv-naive.so", "-lm"])
        lib = ctypes.CDLL('./trisolv-naive.so')
        trisolv_c = lib.kernel_trisolv
        
        trisolv_c.argtypes = [
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float) # A
        ]
    l = np.zeros((n, n)).astype(np.float32)
    x = np.zeros(n).astype(np.float32)
    b = np.zeros(n).astype(np.float32)
    
    
    # init array
    for i in range(n):
         x[i] = -999
         b[i] = i
         for j in range(i+1):
              l[i, j] = (i+n-j+1)*2/n

    l_copy = l.copy()
    x_copy = x.copy()
    b_copy = b.copy()

    perf = timeit(lambda: trisolv(n, l, x, b), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: x", end="", file=sys.stderr)
        for i in range(n):    
            print(f"{x[i]:.2f} ", end="", file=sys.stderr)
            if ((i) % 20) == 0: print("", file=sys.stderr)
        print("\nend   dump: x", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        l_ptr = l_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        x_ptr = x_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: trisolv_c(n, l_ptr, x_ptr, b_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(x_copy[i] - x[i]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "trisolv-naive.so"])