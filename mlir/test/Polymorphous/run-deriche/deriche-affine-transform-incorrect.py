import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile, parallel, reorder, distribute, get_loop, int_attr
from pydsl.type import UInt32, F64, Index, AnyOp, F32
from pydsl.ops import exp, sqrt, pow
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile, CTarget, PolyCTarget
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S
import numpy as np
from timeit import timeit 

Memref2DF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)
Memref1DF32 = MemRefFactory((DYNAMIC,), F32)


def transform_seq(targ: AnyOp):
    pass
    

# 
# m,n,data,corrHe World!!
debug = False
@compile(locals(), transform_seq=transform_seq, dump_mlir=debug, auto_build=(not debug))
def deriche(w: Index, h: Index, alpha: F32, imgIn: Memref2DF32, imgOut: Memref2DF32, y1: Memref2DF32, y2: Memref2DF32, xm: Memref1DF32, tm: Memref1DF32, ym: Memref1DF32, xp: Memref1DF32, tp: Memref1DF32, yp: Memref1DF32, k: Memref1DF32, a: Memref1DF32, b: Memref1DF32, c: Memref1DF32) -> None:
    cst1: F32 = 1.0
    cst2: F32 = 2.0
    zero: F32 = 0.0
    k[am(0)] = (cst1 - exp(-alpha)) * (cst1 - exp((-alpha))) / (cst1 + cst2 * alpha * exp((-alpha)) - exp(cst2 * alpha))
    a[am(1-1)] = k[am(0)]
    a[am(5-1)] = k[am(0)]
    a[am(2-1)] = k[am(0)] * exp((-alpha))*(alpha - cst1)
    a[am(6-1)] = k[am(0)] * exp((-alpha))*(alpha - cst1)
    a[am(3-1)] = k[am(0)] * exp((-alpha))*(alpha + cst1)
    a[am(7-1)] = k[am(0)] * exp((-alpha))*(alpha + cst1)
    a[am(4-1)] = (-k[am(0)]) * exp(-alpha*cst2)
    a[am(8-1)] = (-k[am(0)]) * exp(-alpha*cst2)
    b[am(1-1)] = pow(cst2, (-alpha))
    b[am(2-1)] = -exp(-alpha * cst2)

    for i in arange(w):
        ym[am(0)] = zero
        ym[am(1)] = zero
        xm[am(0)] = zero
        for j in arange(h):
            y1[i, j] = a[am(1-1)] * imgIn[i, j] + a[am(2-1)] * xm[am(0)] + b[am(1-1)] * ym[am(0)] + b[am(2-1)] * ym[am(1)]
            xm[am(0)] = imgIn[i, j]
            ym[am(1)] = ym[am(0)]
            ym[am(0)] = y1[i, j]

    for i in arange(w):
        yp[am(0)] = zero
        yp[am(1)] = zero
        xp[am(0)] = zero
        xp[am(1)] = zero
        for j in arange(h):
            y2[i, h-j-1] = a[am(3-1)] * xp[am(0)] + a[am(4-1)] * xp[am(1)] + b[am(1-1)] * yp[am(0)] + b[am(2-1)] * yp[am(1)]
            xp[am(1)] = xp[am(0)]
            xp[am(0)] = imgIn[i, h-j-1]
            yp[am(1)] = yp[am(0)]
            yp[am(0)] = y2[i, h-j-1]
    
    for i in arange(w):
        for j in arange(h):
            imgOut[i, j] = cst1 * (y1[i, j] + y2[i, j])

    for j in arange(h):
        tm[am(0)] = zero
        ym[am(0)] = zero
        ym[am(1)] = zero
        for i in arange(w):
            y1[i, j] = a[am(5-1)] * imgOut[i, j] + a[am(6-1)] * tm[am(0)] + b[am(1-1)] * ym[am(0)] + b[am(2-1)] * ym[am(1)]
            tm[am(0)] = imgOut[i, j]
            ym[am(1)] = ym[am(0)]
            ym[am(0)] = y1[i, j]
    
    for j in arange(h):
        tp[am(0)] = zero
        tp[am(1)] = zero
        yp[am(0)] = zero
        yp[am(1)] = zero
        for i in arange(w):
            y2[w - i - 1, j] = a[am(7-1)] * tp[am(0)] + a[am(8-1)] * tp[am(1)] + b[am(1-1)] * yp[am(0)] + b[am(2-1)] * yp[am(1)]
            tp[am(1)] = tp[am(0)]
            tp[am(0)] = imgOut[w - i - 1, j]
            yp[am(1)] = yp[am(0)]
            yp[am(0)] = y2[w - i - 1, j]
    
    for i in arange(w):
        for j in arange(h):
            imgOut[i, j] = cst1 * (y1[i, j] + y2[i, j])

if __name__ == "__main__" and not debug:
    w = 192
    h = 128
    
    imgIn = np.zeros((w, h)).astype(np.float32)
    imgOut = np.zeros((w, h)).astype(np.float32)
    y1 = np.zeros((w, h)).astype(np.float32)
    y2 = np.zeros((w, h)).astype(np.float32)
    ym = np.zeros(2).astype(np.float32)
    yp = np.zeros(2).astype(np.float32)
    xp = np.zeros(2).astype(np.float32)
    tp = np.zeros(2).astype(np.float32)
    xm = np.zeros(1).astype(np.float32)
    tm = np.zeros(1).astype(np.float32)
    k = np.zeros(1).astype(np.float32)
    a = np.zeros(8).astype(np.float32)
    b = np.zeros(2).astype(np.float32)
    c = np.zeros(2).astype(np.float32)

    alpha = np.float32(0.25)
    # init array
    for i in range(w):
        for j in range(h):
            imgIn[i, j] = ((313*i+991*j) % 65536) / 65535.0

    perf = timeit(lambda: deriche(w, h, alpha, imgIn, imgOut, y1, y2, xm, tm, ym, xp, tp, yp, k, a, b, c), number=1)
    print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
    print("begin dump: imgOut", end="", file=sys.stderr)
    for i in range(w):    
        for j in range(h):
            if ((i * h + j) % 20) == 0: print("", file=sys.stderr)
            print(f"{imgOut[i, j]:.2f} ", end="", file=sys.stderr)
    print("\nend   dump: imgOut", file=sys.stderr)
    print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)