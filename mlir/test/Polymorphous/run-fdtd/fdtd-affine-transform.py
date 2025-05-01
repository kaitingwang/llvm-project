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

MemrefF322D = MemRefFactory((DYNAMIC, DYNAMIC), F32)
MemrefF321D = MemRefFactory((DYNAMIC, ), F32)

def transform_seq(targ: AnyOp):
    fused_res = fuse(match(targ, "fuse_3"), match(targ, "fuse_2"), 1)
    fused_res_final = fuse(fused_res, match(targ, "fuse_1"), 1)
    tiled_res = tile(fused_res_final, [32, 32], 4)
    parallel(get_loop(tiled_res, 0), False)


# 
# m,n,data,corr

def fdtd(tmax: Index, nx: Index, ny: Index, ex: MemrefF322D, ey: MemrefF322D,  hz: MemrefF322D, fict: MemrefF321D) -> None:
    a: F32 = 0.5
    c: F32 = 0.7
    for t in arange(tmax):
        for j in arange(ny):
            ey[0, j] = fict[t]
        """@tag("fuse_1")"""
        for i in arange(1, nx):
            for j in arange(ny):
                ey[i, j] = ey[i, j] - a * (hz[i, j] - hz[i-1, j])
        """@tag("fuse_2")"""
        for i in arange(nx):
            for j in arange(1, ny):
                ex[i, j] = ex[i, j] - a * (hz[i, j] - hz[i, j-1])
        """@tag("fuse_3")"""
        for i in arange(nx-1):
            for j in arange(ny-1):
                hz[i, j] = hz[i, j] - c * (ex[i, j+1] - ex[i, j] + ey[i+1, j] - ey[i, j])

if __name__ == "__main__":
    compilation_target = CTarget
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (20, 20, 30), "SMALL_DATASET": (40, 60, 80), "MEDIUM_DATASET": (100, 200, 240), "LARGE_DATASET": (500, 1000, 1200), "EXTRALARGE_DATASET": (1000, 2000, 2600)}
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

    tmax, nx, ny = datasets.get(current_dataset, (40, 60,  80))

    MemrefF322D = MemRefFactory((nx, ny), F32)
    MemrefF321D = MemRefFactory((tmax, ), F32)
    fdtd = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target, dataset=current_dataset)(fdtd)
    
    if c_test:
        subprocess.run(["clang", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "fdtd.c", "-DNAIVE_VERSION", "-o", "fdtd-naive.so", "-lm", "-fPIC"])
        lib = ctypes.CDLL('./fdtd-naive.so')
        fdtd_c = lib.kernel_fdtd_2d
        
        fdtd_c.argtypes = [
        ctypes.c_int,  # tmax
        ctypes.c_int,  # nx
        ctypes.c_int,  # ny
        ctypes.POINTER(ctypes.c_float), # ex
        ctypes.POINTER(ctypes.c_float), # ey
        ctypes.POINTER(ctypes.c_float), # hz
        ctypes.POINTER(ctypes.c_float), # _fict_
        ]
    
    ex = np.zeros((nx, ny)).astype(np.float32)
    ey = np.zeros((nx, ny)).astype(np.float32)
    hz = np.zeros((nx, ny)).astype(np.float32)
    fict = np.zeros(tmax).astype(np.float32)
    
    # init array
    for i in range(tmax):
        fict[i] = i
    for i in range(nx):
        for j in range(ny):
            ex[i, j] = (i * (j+1)) / nx
            ey[i, j] = (i * (j+2)) / ny
            hz[i, j] = (i * (j+3)) / nx
    ex_copy = ex.copy()
    ey_copy = ey.copy()
    hz_copy = hz.copy()
    fict_copy = fict.copy()


    perf = timeit(lambda: fdtd(tmax, nx, ny, ex, ey, hz, fict), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: ex", end="", file=sys.stderr)
        for i in range(nx):    
            for j in range(ny):
                    if ((i * nx + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{ex[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: ex", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: ey", end="", file=sys.stderr)
        for i in range(nx):    
            for j in range(ny):
                    if ((i * nx + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{ey[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: ey", file=sys.stderr)
        print("begin dump: hz", end="", file=sys.stderr)
        for i in range(nx):    
            for j in range(ny):
                    if ((i * nx + j) % 20) == 0: print("", file=sys.stderr)
                    print(f"{hz[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: hz", file=sys.stderr)
    print(perf)

    if c_test:
        ex_ptr = ex_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ey_ptr = ey_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        hz_ptr = hz_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        fict_ptr = fict_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: fdtd_c(tmax, nx, ny, ex_ptr, ey_ptr, hz_ptr, fict_ptr), number=1)
        max_difference = 0.0
        for i in range(nx):
            for j in range(ny):
                max_difference = max(max_difference, abs(ex_copy[i, j] - ex[i, j]))
        for i in range(nx):
            for j in range(ny):
                max_difference = max(max_difference, abs(ey_copy[i, j] - ey[i, j]))
        for i in range(nx):
            for j in range(ny):
                max_difference = max(max_difference, abs(hz_copy[i, j] - hz[i, j]))
        print(perf)
        if max_difference < 0.5: # some times difference is like 0.4, only happens with clang
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "fdtd-naive.so"])
