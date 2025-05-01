import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, unroll, get_loop, int_attr, distribute, reorder, parallel
from pydsl.type import UInt32, F32, Index, AnyOp
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile, CTarget, PolyCTarget
from pydsl.affine import    \
    affine_range as arange, \
    affine_map as am,       \
    dimension as D,         \
    symbol as S
import numpy as np
from timeit import timeit

def transform_seq(targ: AnyOp):
    fuse(match(targ, "fuse_1"), match(targ, "fuse_2"), 2)
    tile(match(targ, "fuse"), [8, 32, 32], 6)


MemRefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)

@compile(locals(), dump_mlir=False, transform_seq=transform_seq, auto_build=True)
def jacobi(T: Index, N: Index, a: MemRefF32, b: MemRefF32) -> None:
    """@tag("tile")"""
    for _ in arange(T):
        """@tag("fuse_1")"""
        for i in arange(1, N - 1):
            for j in arange(1, N - 1):
                const:F32 = 0.2
                b[i, j] = (a[i, j] +        \
                                     a[i, j - 1] +    \
                                     a[i, j + 1] +    \
                                     a[i - 1, j] +    \
                                     a[i + 1, j]) * const

        """@tag("fuse_2")"""
        for i in arange(1, N - 1):
            for j in arange(1, N - 1):
                const: F32 = 0.2
                a[i, j] = (b[i, j] +        \
                                     b[i, j - 1] +    \
                                     b[i, j + 1] +    \
                                     b[i - 1, j] +    \
                                     b[i + 1, j]) * const
if __name__ == "__main__":
    tsteps = 40
    n = 90
    
    a = np.zeros((n, n)).astype(np.float32)
    b = np.zeros((n, n)).astype(np.float32)
    
    # init array
    for i in range(n):
        for j in range(n):
            a[i, j] = (i*(j+2) + 2) / n
            b[i, j] = (i*(j+3) + 3) / n
    
    perf = timeit(lambda: jacobi(tsteps, n, a, b), number=1)
    print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
    print("begin dump: A", end="", file=sys.stderr)
    for i in range(n):    
        for j in range(n):
                if ((i * n + j) % 20) == 0: print("", file=sys.stderr)
                print(f"{a[i, j]:.2f} ", end="", file=sys.stderr)
    print("\nend   dump: A", file=sys.stderr)
    print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)