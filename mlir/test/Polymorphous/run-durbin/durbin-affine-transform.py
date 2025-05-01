import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, parallel, reorder, distribute, get_loop, int_attr, unroll
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


MemrefF321D = MemRefFactory((DYNAMIC, ), F32)
MemrefF321 = MemRefFactory((1, ), F32)

def transform_seq(targ: AnyOp):
    fuse(match(targ, "fuse_i1"), match(targ, "fuse_i2"), 1)

    
    
# TODO: Support AllocaOp
# 
# m,n,data,corr

def durbin(n: Index, r: MemrefF321D, y: MemrefF321D, z: MemrefF321D, alpha_mem: MemrefF321, beta_mem: MemrefF321, sum: MemrefF321) -> None:
    a: F32 = 0.5
    b: Index = 0
    c: F32 = 0.7
    zero: F32 = 0.0
    y[am(0)] = zero-r[am(0)]
    beta_mem[am(0)] = F32(1)
    alpha_mem[am(0)] = zero-r[am(0)]

    for k in arange(1, n):
        beta_mem[am(0)] = (F32(1)-alpha_mem[am(0)]*alpha_mem[am(0)])*beta_mem[am(0)]
        sum[am(0)] = zero
        for i in arange(k):
            sum[am(0)] = sum[am(0)] + r[k-i-1]*y[i]
        alpha_mem[am(0)] = zero - (r[k] + sum[am(0)]) / beta_mem[am(0)]
        """@tag("fuse_i1")"""
        for i in arange(k):
            z[i] = y[i] + alpha_mem[am(0)] * y[k-i-1]
        """@tag("fuse_i2")"""
        for i in arange(k):
            y[i] = z[i]

        y[k] = alpha_mem[am(0)]
    

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
                print("Durbin Poly Target cannot be supported due to missing alloca op support")
                exit()
            case "CTEST":
                c_test = True
            case "NOCTEST":
                c_test = False
            case "DUMP_ARRAY":
                output_array = True
        if datasets.get(arg) is not None:
            current_dataset = arg
    
    durbin = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target)(durbin)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "durbin.c", "-DNAIVE_VERSION", "-o", "durbin-naive.so", "-lm"])
        lib = ctypes.CDLL('./durbin-naive.so')
        durbin_c = lib.kernel_durbin
        
        durbin_c.argtypes = [
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # r
        ctypes.POINTER(ctypes.c_float), # y
        ]

    n = datasets.get(current_dataset, 120)
    
    r = np.zeros(n).astype(np.float32)
    y = np.zeros(n).astype(np.float32)
    z = np.zeros(n).astype(np.float32)
    alpha = np.zeros(1).astype(np.float32)
    beta = np.zeros(1).astype(np.float32)
    sum = np.zeros(1).astype(np.float32)

    
    # init array
    for i in range(n):
        r[i] = (n+1-i)
    
    r_copy = r.copy()
    y_copy = y.copy()

    perf = timeit(lambda: durbin(n, r, y, z, alpha, beta, sum), number=1)
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
        r_ptr = r_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y_ptr = y_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: durbin_c(n, r_ptr, y_ptr), number=1)
        max_difference = 0.0
        for i in range(n):
            max_difference = max(max_difference, abs(y_copy[i] - y[i]))
        print(perf)
        if max_difference == 0.0:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "durbin-naive.so"])
