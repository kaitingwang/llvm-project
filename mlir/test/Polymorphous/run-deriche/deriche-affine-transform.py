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
import subprocess
import ctypes

Memref2DF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)
Memref1DF32 = MemRefFactory((DYNAMIC,), F32)


def transform_seq(targ: AnyOp):
    pass
    
# TODO: support Alloca op so that array expansion can be used
# 
# m,n,data,corr
def deriche(w: Index, h: Index, alpha: F32, imgIn: Memref2DF32, imgOut: Memref2DF32, y1: Memref2DF32, y2: Memref2DF32, xm: Memref1DF32, tm: Memref1DF32, ym: Memref1DF32, xp: Memref1DF32, tp: Memref1DF32, yp: Memref1DF32, k: Memref1DF32, a: Memref1DF32, b: Memref1DF32, c: Memref1DF32) -> None:
    cst1: F32 = 1.0
    cst2: F32 = 2.0
    zero: F32 = 0.0
    k[0] = (cst1 - exp(-alpha)) * (cst1 - exp((-alpha))) / (cst1 + cst2 * alpha * exp((-alpha)) - exp(cst2 * alpha))
    a[1-1] = k[0]
    a[5-1] = k[0]
    a[2-1] = k[0] * exp((-alpha))*(alpha - cst1)
    a[6-1] = k[0] * exp((-alpha))*(alpha - cst1)
    a[3-1] = k[0] * exp((-alpha))*(alpha + cst1)
    a[7-1] = k[0] * exp((-alpha))*(alpha + cst1)
    a[4-1] = (-k[0]) * exp(-alpha*cst2)
    a[8-1] = (-k[0]) * exp(-alpha*cst2)
    b[1-1] = pow(cst2, (-alpha))
    b[2-1] = -exp(-alpha * cst2)

    for i in arange(w):
        ym[0] = zero
        ym[1] = zero
        xm[0] = zero
        for j in arange(h):
            y1[i, j] = a[1-1] * imgIn[i, j] + a[2-1] * xm[0] + b[1-1] * ym[0] + b[2-1] * ym[1]
            xm[0] = imgIn[i, j]
            ym[1] = ym[0]
            ym[0] = y1[i, j]

    for i in arange(w):
        yp[0] = zero
        yp[1] = zero
        xp[0] = zero
        xp[1] = zero
        for j in arange(h):
            y2[i, h-j-1] = a[3-1] * xp[0] + a[4-1] * xp[1] + b[1-1] * yp[0] + b[2-1] * yp[1]
            xp[1] = xp[0]
            xp[0] = imgIn[i, h-j-1]
            yp[1] = yp[0]
            yp[0] = y2[i, h-j-1]
    
    for i in arange(w):
        for j in arange(h):
            imgOut[i, j] = cst1 * (y1[i, j] + y2[i, j])

    for j in arange(h):
        tm[0] = zero
        ym[0] = zero
        ym[1] = zero
        for i in arange(w):
            y1[i, j] = a[5-1] * imgOut[i, j] + a[6-1] * tm[0] + b[1-1] * ym[0] + b[2-1] * ym[1]
            tm[0] = imgOut[i, j]
            ym[1] = ym[0]
            ym[0] = y1[i, j]
    
    for j in arange(h):
        tp[0] = zero
        tp[1] = zero
        yp[0] = zero
        yp[1] = zero
        for i in arange(w):
            y2[w - i - 1, j] = a[7-1] * tp[0] + a[8-1] * tp[1] + b[1-1] * yp[0] + b[2-1] * yp[1]
            tp[1] = tp[0]
            tp[0] = imgOut[w - i - 1, j]
            yp[1] = yp[0]
            yp[0] = y2[w - i - 1, j]
    
    for i in arange(w):
        for j in arange(h):
            imgOut[i, j] = cst1 * (y1[i, j] + y2[i, j])

if __name__ == "__main__":
    c_test = False
    compilation_target = CTarget
    datasets = {"MINI_DATASET": (64, 64), "SMALL_DATASET": (192, 128), "MEDIUM_DATASET": (720, 480), "LARGE_DATASET": (4096, 2160), "EXTRALARGE_DATASET": (7860, 4320)}
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
    
    deriche = compile(locals(), transform_seq=transform_seq, dump_mlir=False, auto_build=True, target_class=compilation_target)(deriche)

    if c_test:
        subprocess.run(["gcc-13", "-std=c11", "-O3", "-DPOLYBENCH_TIME", "-DDATA_TYPE_IS_FLOAT", "../utilities/polybench.c", "-shared", f"-D{current_dataset}", "-I../utilities/", "deriche.c", "-DNAIVE_VERSION", "-o", "deriche-naive.so", "-lm"])
        lib = ctypes.CDLL('./deriche-naive.so')
        deriche_c = lib.kernel_deriche
        
        deriche_c.argtypes = [
        ctypes.c_int,  # w
        ctypes.c_int,  # h
        ctypes.c_float,# alpha
        ctypes.POINTER(ctypes.c_float), # imgIn
        ctypes.POINTER(ctypes.c_float), # imgOut
        ctypes.POINTER(ctypes.c_float), # y1
        ctypes.POINTER(ctypes.c_float), # y2
        ]

    w, h = datasets.get(current_dataset, (192, 128))
    
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
        
    imgIn_copy = imgIn.copy()
    imgOut_copy = imgOut.copy()
    y1_copy = y1.copy()
    y2_copy = y2.copy()

    perf = timeit(lambda: deriche(w, h, alpha, imgIn, imgOut, y1, y2, xm, tm, ym, xp, tp, yp, k, a, b, c), number=1)
    if output_array:
        print("==BEGIN DUMP_ARRAYS==", file=sys.stderr)
        print("begin dump: imgOut", end="", file=sys.stderr)
        for i in range(w):    
            for j in range(h):
                if ((i * h + j) % 20) == 0: print("", file=sys.stderr)
                print(f"{imgOut[i, j]:.2f} ", end="", file=sys.stderr)
        print("\nend   dump: imgOut", file=sys.stderr)
        print("==END   DUMP_ARRAYS==", file=sys.stderr)
    print(perf)

    if c_test:
        imgIn_ptr = imgIn_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        imgOut_ptr = imgOut_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y1_ptr = y1_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        y2_ptr = y2_copy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        perf = timeit(lambda: deriche_c(w, h, alpha, imgIn_ptr, imgOut_ptr, y1_ptr, y2_ptr), number=1)
        max_difference = 0.0
        for i in range(w):
            for j in range(h):
                max_difference = max(max_difference, abs(imgOut_copy[i, j] - imgOut[i, j]))
        print(perf)
        if max_difference < 0.001:
            print("results are correct.")
        else:
            print(f"results incorrect! Max value difference is {max_difference:2f}")
            exit(1)
        subprocess.run(["rm", "deriche-naive.so"])