import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, parallel, reorder, distribute, get_loop, int_attr, unroll
from pydsl.type import SInt32 as i32, SInt8 as i8, Index, AnyOp, F32, Bool
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile, CTarget, PolyCTarget
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S,                    \
    integer_set as iset
import numpy as np
from timeit import timeit
import ctypes
import subprocess

MemrefF322D = MemRefFactory((DYNAMIC, DYNAMIC), i32)
MemrefF321D = MemRefFactory((DYNAMIC, ), i8)

def transform_seq(targ: AnyOp):
    tile(match(targ, "tile"), [32, 32], 4)
    
    
# max_score(a, b) = a if a >= b else b
# match(b1, b2) = i32(1) if (b1 + b2) == i8(3) else i32(0)
# m,n,data,corr

def nussinov(n: Index, seq: MemrefF321D, table: MemrefF322D) -> None:
    a: F32 = 1.0
    """@tag("tile")"""
    for i in arange(n):
        for j in arange(n-i, n):
            if iset(j - 1 >= 0):
                table[n-i-1, j] = table[n-i-1, j] if table[n-i-1, j] >= table[n-i-1, j-1] else table[n-i-1, j-1]
            if iset(i >= 1):
                 table[n-i-1, j] = table[n-i-1, j] if table[n-i-1, j] >= table[n-i-1+1, j] else table[n-i-1+1, j]
            if iset(j - 1 >= 0 and n-i < n):
                if iset(n-i-1 < j - 1):
                    table[n-i-1, j] = table[n-i-1, j] if table[n-i-1, j] >= (table[n-i-1+1, j-1]+(i32(1) if (seq[n-i-1] + seq[j]) == i8(3) else i32(0))) else (table[n-i-1+1, j-1]+(i32(1) if (seq[n-i-1] + seq[j]) == i8(3) else i32(0)))
                else:
                    table[n-i-1, j] = table[n-i-1, j] if table[n-i-1, j] >= table[n-i-1+1, j-1] else table[n-i-1+1, j-1]
            for k in arange(n-i, j):
                table[n-i-1, j] = table[n-i-1, j] if table[n-i-1, j] >= (table[n-i-1, k] + table[k+1, j]) else (table[n-i-1, k] + table[k+1, j])
                

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": 60, "SMALL_DATASET": 180, "MEDIUM_DATASET": 500, "LARGE_DATASET": 2500, "EXTRALARGE_DATASET": 5500}
    current_dataset = "SMALL_DATASET"
    output_array = False
    for arg in sys.argv:
        match arg:
            case "CTARGET":
                compilation_target = CTarget
            case "POLYCTARGET":
                compilation_target = PolyCTarget
                print("Nussinov Poly Target cannot be supported due to missing helper function support")
                exit()
            case "CTEST":
                c_test = True
            case "NOCTEST":
                c_test = False
            case "DUMP_ARRAY":
                output_array = True
        if datasets.get(arg) is not None:
            current_dataset = arg
    
    n = datasets.get(current_dataset, 180)

    MemrefF322D = MemRefFactory((n, n), i32)
    MemrefF321D = MemRefFactory((n, ), i8)

    nussinov = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(nussinov)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_INT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "nussinov.c", "-DNAIVE_VERSION", "-o", "nussinov-naive.so", "-lm"])
        lib = ctypes.CDLL('./nussinov-naive.so')
        nussinov_c = lib.kernel_nussinov
        
        nussinov_c.argtypes = [
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_int8), # seq
        ctypes.POINTER(ctypes.c_int32), # table
        ]
    
    table = np.zeros((n, n)).astype(np.int32)
    seq = np.zeros(n).astype(np.int8)
    
    #init array
    for i in range(n):
        seq[i] = (i+1) % 4
    
    seq_copy = seq.copy()
    table_copy = table.copy()

    perf = timeit(lambda: nussinov(n, seq, table), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: table", end="", file=sys.stderr)
        t=0
        for i in range(n):    
            for j in range(i, n):
                    if ((t) % 20) == 0: print("", file=sys.stderr)
                    print(f"{table[i, j]:.2f} ", end="", file=sys.stderr)
                    t+=1
        print("\nend   dump: table", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        seq_ptr = seq_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        table_ptr = table_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        perf = timeit(lambda: nussinov_c(n, seq_ptr, table_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            for j in range(n):
                max_difference = max(max_difference, abs(table_copy[i, j] - table[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "nussinov-naive.so"])