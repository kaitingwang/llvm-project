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
import ctypes
import subprocess

MemrefF32MN = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF32NN = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC, DYNAMIC), F32)

def transform_seq(targ: AnyOp):
    distribute_result = distribute(match(targ, 'distribute'), 2)
    tile_result = tile(get_loop(distribute_result, 1), [32, 32], 7)
    reorder(get_loop(tile_result, 5), get_loop(tile_result, 6))
    reorder(get_loop(tile_result, 2), get_loop(tile_result, 3))
    parallel(get_loop(tile_result, 1))


def gramschmidt(m: Index, n: Index, A: MemrefF32MN, R: MemrefF32NN, Q: MemrefF32MN, temp: MemrefF321D) -> None:
    b: F32 = 0.0
    for k in arange(n):
        temp[0] = b
        for i in arange(m):
            temp[0] = temp[0] + A[i, k] * A[i, k]
        R[k, k] = sqrt(temp[0])
        for i in arange(m):
             Q[i, k] = A[i, k] / R[k, k]
        """@tag("distribute")"""
        for j in arange(k+1, n):
            """@int_attr("set", 0)"""
            R[k, j] = b
            for i in arange(m):
                with recursively(lambda x: int_attr(x, "set", 1)):
                    R[k, j] = R[k, j] + Q[i, k] * A[i, j]
            for i in arange(m):
                with recursively(lambda x: int_attr(x, "set", 1)):
                    A[i, j] = A[i, j] - Q[i, k] * R[k, j]

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (20, 30), "SMALL_DATASET": (60, 80), "MEDIUM_DATASET": (200, 240), "LARGE_DATASET": (1000, 1200), "EXTRALARGE_DATASET": (2000, 2600)}
    current_dataset = "SMALL_DATASET"
    output_array = False
    for arg in sys.argv:
        match arg:
            case "CTARGET":
                compilation_target = CTarget
            case "POLYCTARGET":
                compilation_target = PolyCTarget
                print("Deriche Poly Target cannot be supported due to missing alloca op support")
                exit()
            case "CTEST":
                c_test = True
            case "NOCTEST":
                c_test = False
            case "DUMP_ARRAY":
                output_array = True
        if datasets.get(arg) is not None:
            current_dataset = arg

    m, n = datasets.get(current_dataset, (60, 80))

    MemrefF32MN = MemRefFactory((m, n), F32)
    MemrefF32NN = MemRefFactory((n, n), F32)
    MemrefF321D = MemRefFactory((n,), F32)
    gramschmidt = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(gramschmidt)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "gramschmidt.c", "-DNAIVE_VERSION", "-o", "gramschmidt-naive.so", "-lm"])
        lib = ctypes.CDLL('./gramschmidt-naive.so')
        gramschmidt_c = lib.kernel_gramschmidt
        
        gramschmidt_c.argtypes = [
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.POINTER(ctypes.c_float), # a
        ctypes.POINTER(ctypes.c_float), # r
        ctypes.POINTER(ctypes.c_float), # q
        ]
    
    a = np.zeros((m, n)).astype(np.float32)
    q = np.zeros((m, n)).astype(np.float32)
    r = np.zeros((n, n)).astype(np.float32)
    t = np.zeros(n).astype(np.float32)
    
    # init array
    for i in range(m):
        for j in range(n):
             a[i, j] = ((((i*j) % m) / m)*100)+10
    
    a_copy = a.copy()
    q_copy = q.copy()
    r_copy = r.copy()

    perf = timeit(lambda: gramschmidt(m, n, a, r, q, t), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: R", end="", file=sys.stderr)
        for i in range(n):    
            for j in range(n):
                    if ((i * n + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{r[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: R", file=sys.stderr)
        print("begin dump: Q", end="", file=sys.stderr)
        for i in range(m):    
            for j in range(n):
                    if ((i * n + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{q[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: Q", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        a_ptr = a_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        r_ptr = r_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        q_ptr = q_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: gramschmidt_c(m, n, a_ptr, r_ptr, q_ptr), number=1)
        max_difference = 0.0
        for i in range(m):
            for j in range(n):
                max_difference = max(max_difference, abs(q_copy[i, j] - q[i, j]))
        for i in range(n):
            for j in range(n):
                max_difference = max(max_difference, abs(r_copy[i, j] - r[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "gramschmidt-naive.so"])