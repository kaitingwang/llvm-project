from statistics import mean, stdev
import sys
import time
from timeit import timeit
from typing import Tuple

import numpy as np

from pydsl.transform import fuse, get_loop, tile, parallel, tag, match_tag as match
from pydsl.type import Index, AnyOp, F32
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile, CTarget, PolyCTarget
from pydsl.affine import affine_range as arange

N = 20
MemF32 = MemRefFactory((DYNAMIC, DYNAMIC, DYNAMIC), F32)


# ---------------------------------------------------- Fuse+Tile ----------------------------------------------------
def transform_seq_fuse_tile(targ: AnyOp):
    fuse(match(targ, "fuse_1"), match(targ, "fuse_2"), 3)
    tile(match(targ, "tile"), [8, 8, 32, 32], 8)
    

def heat_fuse_tile(tsteps: Index, n: Index, A: MemF32, B: MemF32):
    a: F32 = 2.0
    b: F32 = 0.125
    """@tag("tile")"""
    for _ in arange(tsteps):
        """@tag("fuse_1")"""
        for i in arange(1, n-1):
            for j in arange(1, n-1):
                for k in arange(1, n-1):
                    B[i,j,k] = A[i,j,k] + b * (
                        A[i+1,j,k] - a * A[i,j,k] + A[i-1,j,k] + \
                        A[i,j+1,k] - a * A[i,j,k] + A[i,j-1,k] + \
                        A[i,j,k+1] - a * A[i,j,k] + A[i,j,k-1]
                    )
        """@tag("fuse_2")"""
        for i in arange(1, n-1):
            for j in arange(1, n-1):
                for k in arange(1, n-1):
                    A[i,j,k] = B[i,j,k] + b * (
                        B[i+1,j,k] - a * B[i,j,k] + B[i-1,j,k] + \
                        B[i,j+1,k] - a * B[i,j,k] + B[i,j-1,k] + \
                        B[i,j,k+1] - a * B[i,j,k] + B[i,j,k-1]
                    )


# ---------------------------------------------------- Parallel ----------------------------------------------------
def transform_seq_parallel(targ: AnyOp):
    parallel(match(targ, "parallel1"))
    parallel(match(targ, "parallel2"))


def heat_parallel(tsteps: Index, n: Index, A: MemF32, B: MemF32):
    a: F32 = 2.0
    b: F32 = 0.125
    for _ in arange(tsteps):
        """@tag("parallel1")"""
        for i in arange(1, n-1):
            for j in arange(1, n-1):
                for k in arange(1, n-1):
                    B[i,j,k] = A[i,j,k] + b * (
                        A[i+1,j,k] - a * A[i,j,k] + A[i-1,j,k] + \
                        A[i,j+1,k] - a * A[i,j,k] + A[i,j-1,k] + \
                        A[i,j,k+1] - a * A[i,j,k] + A[i,j,k-1]
                    )
        """@tag("parallel2")"""
        for i in arange(1, n-1):
            for j in arange(1, n-1):
                for k in arange(1, n-1):
                    A[i,j,k] = B[i,j,k] + b * (
                        B[i+1,j,k] - a * B[i,j,k] + B[i-1,j,k] + \
                        B[i,j+1,k] - a * B[i,j,k] + B[i,j-1,k] + \
                        B[i,j,k+1] - a * B[i,j,k] + B[i,j,k-1]
                    )

# ---------------------------------------------------- Running ----------------------------------------------------

def check_correctness(func, dump_file_name="python.log"):
    n = N

    A = np.fromfunction(lambda i, j, k: (i + j + (n-k))* 10 / (n), (n, n, n), dtype=np.float32)
    B = np.empty_like(A)
    B[:] = A

    func(40, n, A, B)

    file = open(dump_file_name, "w")
    file.write("==BEGIN DUMP_ARRAYS==\n")
    file.write("begin dump: A")
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                if (i * n * n + j * n + k) % 20 == 0:
                    file.write("\n")
                file.write(f"{A[i][j][k]:.2f} ")
    file.write("\nend   dump: A\n")
    file.write("==END   DUMP_ARRAYS==\n")


def measure_performance(func):
    n = N
    
    A = np.fromfunction(lambda i, j, k: (i + j + (n-k))* 10 / (n), (n, n, n), dtype=np.float32)
    B = np.empty_like(A)
    B[:] = A

    perf = timeit(lambda: func(500, n, A, B), number=1)
    
    print(f"\n", file=sys.stdout)
    print(f"Heat 3D running with LARGE DATASET took {perf:.6f} seconds.", file=sys.stdout)
    print(f"\n", file=sys.stdout)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python heat.py debug|perf")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "debug":
        N = 20
        MemF32 = MemRefFactory((N, N, N), F32)
        heat_slow = compile(locals(), transform_seq=None, dump_mlir=False, auto_build=True, target_class=CTarget)(heat_fuse_tile)
        heat_fast = compile(locals(), transform_seq=transform_seq_fuse_tile, dump_mlir=False, auto_build=True, target_class=PolyCTarget)(heat_fuse_tile)
        heat_faster = compile(locals(), transform_seq=transform_seq_parallel, dump_mlir=False, auto_build=True, target_class=PolyCTarget)(heat_parallel)

        print(f"ensure algorithmic correctness\n")
        check_correctness(heat_slow, "python_naive.log")
        check_correctness(heat_fast, "python_opt1.log")
        check_correctness(heat_faster, "python_opt2.log")
    elif arg == "perf":
        N = 120
        MemF32 = MemRefFactory((N, N, N), F32)
        heat_slow = compile(locals(), transform_seq=None, dump_mlir=False, auto_build=True, target_class=CTarget)(heat_fuse_tile)
        heat_fast = compile(locals(), transform_seq=transform_seq_fuse_tile, dump_mlir=False, auto_build=True, target_class=PolyCTarget)(heat_fuse_tile)
        heat_faster = compile(locals(), transform_seq=transform_seq_parallel, dump_mlir=False, auto_build=True, target_class=PolyCTarget)(heat_parallel)

        print(f"running heat slow\n")
        measure_performance(heat_slow)
        print(f"running heat fast\n")
        measure_performance(heat_fast)
        print(f"running heat faster\n")
        measure_performance(heat_faster)
    else:
        print("Usage: python heat.py debug|perf")
        sys.exit(1)
