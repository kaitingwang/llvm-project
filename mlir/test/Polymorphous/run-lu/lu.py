from statistics import mean, stdev
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, reorder, get_loop
from pydsl.type import UInt32, F32, Index, AnyOp
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile, CTarget, PolyCTarget
from pydsl.affine import            \
    affine_range as arange
import numpy as np
from timeit import timeit

MemrefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)

N = 2000

def init_array(n, a):
    # init array
    for i in range(n):
        for j in range(i+1):
            a[i, j] = -(j % n) / n + 1
        for j in range(i+1, n):
            a[i, j] = 0.0
        a[i, i] = 1.0

    # make the matrix positive semmi-definite
    b = np.zeros((n, n)).astype(np.float32)
    # for r in range(n):
    #     for s in range(n):
    #         b[r, s] = 0.0
    for t in range(n):
        for r in range(n):
            for s in range(n):
                b[r, s] += a[r, t] * a[s, t]
    for r in range(n):
        for s in range(n):
            a[r, s] = b[r, s]


def transform_seq(targ: AnyOp):
    fuse_into(
    fuse_into(
    fuse_into(
    fuse_into(
        match(targ, 'fuse_1'), 
        match(targ, 'fuse_target1')),
        match(targ, 'fuse_target2')), 
        match(targ, 'fuse_target3')), 
        match(targ, 'fuse_target4'))

    fuse(match(targ, 'fuse_4'), match(targ, 'fuse_3'), 2)

    tiled_loops = tile(match(targ, 'tile'), [32, 32, 32], 6)
    reorder(get_loop(tiled_loops, 4), get_loop(tiled_loops, 5))


def lu_python(v0: Index, arg1: MemrefF32) -> None:
    """@tag("tile")"""
    for arg2 in arange(v0):

        """@tag("fuse_4")"""
        for arg3 in arange(arg2):

            """@tag("fuse_1")"""
            for arg4 in arange(arg3):
                arg1[arg2, arg3] =    \
                    arg1[arg2, arg3]  \
                    - (arg1[arg2, arg4] 
                    * arg1[arg4, arg3])
            
            """@tag("fuse_target1")"""
            v1 = arg1[arg3, arg3]

            """@tag("fuse_target2")"""
            v2 = arg1[arg2, arg3]

            """@tag("fuse_target3")"""
            v3 = v2 / v1

            """@tag("fuse_target4")"""
            arg1[arg2, arg3] = v3

        """@tag("fuse_3")"""
        for arg3 in arange(arg2, v0):
            for arg4 in arange(arg2):
                arg1[arg2, arg3] =    \
                    arg1[arg2, arg3]  \
                    - (arg1[arg2, arg4] 
                    * arg1[arg4, arg3])

    return


# ---------------------------------------------------- Running ----------------------------------------------------

def check_correctness(func, dump_file_name="python.log"):
    n = N
    a = np.zeros((N, N)).astype(np.float32)
    
    init_array(n, a)

    func(n, a)

    file = open(dump_file_name, "w")
    print("==BEGIN DUMP_ARRAYS==", file=file)
    print("begin dump: A", end="", file=file)
    for i in range(n):
        for j in range(n):
            if (i * n + j) % 20 == 0: print("", file=file)
            print(f"{a[i, j]:.2f} ", end="", file=file)
    print("\nend   dump: A", file=file)
    print("==END   DUMP_ARRAYS==", file=file)


def measure_performance(func):
    n = N
    a = np.zeros((N, N)).astype(np.float32)
    
    # DO NOT initialize array under large shapes, it takes soooo long
    # init_array(n, a)

    perf = timeit(lambda: func(n, a), number=1)
    
    print(f"\n", file=sys.stdout)
    print(f"LU running with LARGE DATASET took {perf:.6f} seconds.", file=sys.stdout)
    print(f"\n", file=sys.stdout)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lu.py debug|perf")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "debug":
        N = 120
        MemrefF32 = MemRefFactory((N, N), F32)
        lu_slow = compile(locals(), transform_seq=None, dump_mlir=False, auto_build=True, target_class=CTarget)(lu_python)
        lu_fast = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=PolyCTarget)(lu_python)

        print(f"ensure algorithmic correctness\n")
        check_correctness(lu_slow, "python_naive.log")
        check_correctness(lu_fast, "python_opt.log")
    elif arg == "perf":
        N = 2000
        MemrefF32 = MemRefFactory((N, N), F32)
        lu_slow = compile(locals(), transform_seq=None, dump_mlir=False, auto_build=True, target_class=CTarget)(lu_python)
        lu_fast = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=PolyCTarget)(lu_python)

        print(f"running lu slow\n")
        measure_performance(lu_slow)
        print(f"running lu fast\n")
        measure_performance(lu_fast)
    else:
        print("Usage: python lu.py debug|perf")
        sys.exit(1)
