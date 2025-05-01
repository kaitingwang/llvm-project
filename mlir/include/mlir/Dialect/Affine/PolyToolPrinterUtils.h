//===- PolyToolPrinterUtils.h - PolyTool printer utlities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares a set of utilities for PolyTool to generate the C
// code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_POLYTOOLPRINTERUTILS_H
#define MLIR_DIALECT_AFFINE_POLYTOOLPRINTERUTILS_H

#include "TG.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include <unistd.h>



// MLIR is in SSA form, while this is useful for
// compiling down to assembly, it makes C generation
// annoying (especially if you want it to be readable)
// To get around this, we employ the following algorithm
// 1. Maintain a map from MLIR SSA Values => strings
// 2. For all variables and function arguments, map
//    the MLIR value to the corresponding variable name
// 3. For all values derived from these (IE, expressions)
//    map the MLIR SSA value to the C-string which computes
//    it

struct UnparserState {
    // polytool fields
    mlir::func::FuncOp &targetFunc;
    llvm::DenseMap<mlir::Operation *, mlir::affine::StatementInfo> &statementInfoMap;
    mlir::affine::Node *tgRoot;
    std::string codegenFileName;

    // cloog temp file name
    std::string cloogFileName;


    // A boolean which lets us know if we should generate code in the polybench style
    bool polybenchCodegen;
    // the name of the function we are un-parsing
    std::string benchmarkName;
    // a map from mlir::Value strucs to their variable name in the code
    // It is used when generating code involving variables
    llvm::DenseMap<mlir::Value, std::string> valueToName;
    
    // Contains imports and macro definitions
    std::string headers;

    // Contains variable definitions
    size_t varCount = 0;
    std::string varDefs;

    // Statement Definitions
    size_t statementCount = 0;
    std::string statementDefs;

    // contains the function signature
    std::string signature;

    // contains the loop structures
    std::string loopStructure;

    // scattering dimension size
    size_t scatteringDimension = 0;

    // mapping between SSA statement number and its Static Control Part
    llvm::SmallVector<size_t> ssaToSCoPMap;

    std::string declget(const mlir::Value &v);
    void defineStatement(std::string statementBody, int numContainingLoops);

    UnparserState(
        mlir::func::FuncOp &targetFunc,
        llvm::DenseMap<mlir::Operation *,
        mlir::affine::StatementInfo> &statementInfoMap,
        mlir::affine::Node *tgRoot,
        std::string codegenFileName,
        bool polybenchCodegen)
        : targetFunc(targetFunc),
        statementInfoMap(statementInfoMap),
        tgRoot(tgRoot),
        codegenFileName(codegenFileName),
        polybenchCodegen(polybenchCodegen) {
          this->cloogFileName =  "/tmp/validator_scop_out." + std::to_string(getpid()) + ".cloog";
        }
};

std::unordered_map<std::string, std::string> predefinedKernels = {
    {"lu", "void lu(int arg1, DATA_TYPE POLYBENCH_2D(arg2,N,N,arg1,arg1))"}, 
    {"seidal", "void seidal(int arg1, int arg2, DATA_TYPE POLYBENCH_2D(arg3,N,N,arg2,arg2))"},
    {"cholesky", "void cholesky(int arg1, DATA_TYPE POLYBENCH_2D(arg2,N,N,arg1,arg1))"},
    {"jacobi1d", "void jacobi1d(int arg1, int arg2, DATA_TYPE POLYBENCH_1D(arg3, N, arg2), DATA_TYPE POLYBENCH_1D(arg4, N, arg2))"},
    {"jacobi2d", "void jacobi2d(int arg1, int arg2, DATA_TYPE POLYBENCH_2D(arg3, N, N, arg2, arg2), DATA_TYPE POLYBENCH_2D(arg4, N, N, arg2, arg2))"},
    {"fdtd", "void fdtd(int arg1, int arg2, int arg3, DATA_TYPE POLYBENCH_2D(arg4, NX, NY, arg2, arg3), DATA_TYPE POLYBENCH_2D(arg5, NX, NY, arg2, arg3), DATA_TYPE POLYBENCH_2D(arg6, NX, NY, arg2, arg3), DATA_TYPE POLYBENCH_1D(arg7, TMAX, arg1))"},
    {"mvt", "void mvt(int arg1, DATA_TYPE POLYBENCH_1D(arg2,N,arg1), DATA_TYPE POLYBENCH_1D(arg3,N,arg2), DATA_TYPE POLYBENCH_1D(arg4,N,arg3), DATA_TYPE POLYBENCH_1D(arg5,N,arg4), DATA_TYPE POLYBENCH_2D(arg6,N,N,arg5,arg5))"}, 
    {"heat", "void heat(int arg1, int arg2, DATA_TYPE POLYBENCH_3D(arg3, N, N, N, arg2, arg2, arg2), DATA_TYPE POLYBENCH_3D(arg4, N, N, N, arg2, arg2, arg2))"},
    {"gramschmidt", "void gramschmidt(int arg1, int arg2, DATA_TYPE POLYBENCH_2D(arg3,M,N,arg1,arg2), DATA_TYPE POLYBENCH_2D(arg4,N,N,arg2,arg2), DATA_TYPE POLYBENCH_2D(arg5,M,N,arg1,arg2))"},
    {"correlation", "void correlation(int arg1, int arg2, DATA_TYPE arg3, DATA_TYPE POLYBENCH_2D(arg4,N,M,arg2,arg1), DATA_TYPE POLYBENCH_2D(arg5,M,M,arg1,arg1), DATA_TYPE POLYBENCH_1D(arg6,M,arg1), DATA_TYPE POLYBENCH_1D(arg7,M,arg1))"},
    {"bicg", "void bicg(int arg1, int arg2, DATA_TYPE POLYBENCH_2D(arg3,N,M,arg1,arg1), DATA_TYPE POLYBENCH_1D(arg4,M,arg2), DATA_TYPE POLYBENCH_1D(arg5,N,arg3), DATA_TYPE POLYBENCH_1D(arg6,M,arg4), DATA_TYPE POLYBENCH_1D(arg7,N,arg5))"},
    {"atax", "void atax(int arg1, int arg2, DATA_TYPE POLYBENCH_2D(arg3,M,N,arg1,arg2), DATA_TYPE POLYBENCH_1D(arg4,N,arg2), DATA_TYPE POLYBENCH_1D(arg5,N,arg2), DATA_TYPE POLYBENCH_1D(arg6,M,arg1))"},
    {"trisolv", "void trisolv(int arg1, DATA_TYPE POLYBENCH_2D(arg2,N,N,arg1,arg1), DATA_TYPE POLYBENCH_1D(arg3,N,arg1), DATA_TYPE POLYBENCH_1D(arg4,N,arg1))"},
    {"ludcmp", "void ludcmp(int arg1, DATA_TYPE POLYBENCH_2D(arg2,N,N,arg1,arg1), DATA_TYPE POLYBENCH_1D(arg3,N,arg1), DATA_TYPE POLYBENCH_1D(arg4,N,arg1), DATA_TYPE POLYBENCH_1D(arg5,N,arg1))"}, 
    {"doitgen", "void doitgen(int arg1, int arg2, int arg3, DATA_TYPE POLYBENCH_3D(arg4,NR,NQ,NP,arg1,arg2,arg3), DATA_TYPE POLYBENCH_2D(arg5,NP,NP,arg3,arg3), DATA_TYPE POLYBENCH_1D(arg6,NP,arg3))"},
    {"covariance", "void covariance(int arg1, int arg2, DATA_TYPE arg3, DATA_TYPE POLYBENCH_2D(arg4,N,M,n,m), DATA_TYPE POLYBENCH_2D(arg5,M,M,m,m), DATA_TYPE POLYBENCH_1D(arg6,M,m))"},
    {"durbin", "void durbin(int arg1, DATA_TYPE POLYBENCH_1D(arg2,N,arg1), DATA_TYPE POLYBENCH_1D(arg3,N,arg2))"},
    {"deriche", "void deriche(int arg1, int arg2, DATA_TYPE arg3, DATA_TYPE POLYBENCH_2D(arg4, W, H, arg1, arg2), DATA_TYPE POLYBENCH_2D(arg5, W, H, arg1, arg2), DATA_TYPE POLYBENCH_2D(arg6, W, H, arg1, arg2), DATA_TYPE POLYBENCH_2D(arg7, W, H, arg1, arg2))"},
    {"nussinov", "void nussinov(int arg1, char POLYBENCH_1D(arg2,N,arg1), DATA_TYPE POLYBENCH_2D(arg3,N,N,arg2,arg2))"},
    {"adi", "void adi(int arg1, int arg2, DATA_TYPE POLYBENCH_2D(arg3,N,N,arg2,arg2), DATA_TYPE POLYBENCH_2D(arg4,N,N,arg2,arg2), DATA_TYPE POLYBENCH_2D(arg5,N,N,arg2,arg2), DATA_TYPE POLYBENCH_2D(arg6,N,N,arg2,arg2))"},
    {"floyd_warshall", "void floyd_warshall(int arg1, DATA_TYPE POLYBENCH_2D(arg2,N,N,arg1,arg1))"},
    // GEMM and his good friends~
    {"gemm", "void gemm(int arg1, int arg2, int arg3, DATA_TYPE arg4, DATA_TYPE arg5, DATA_TYPE POLYBENCH_2D(arg6,NI,NJ,arg1,arg2), DATA_TYPE POLYBENCH_2D(arg7,NI,NK,arg1,arg3), DATA_TYPE POLYBENCH_2D(arg8,NK,NJ,arg3,arg2))"},
    {"syrk", "void syrk(int arg1, int arg2, DATA_TYPE arg3, DATA_TYPE arg4, DATA_TYPE POLYBENCH_2D(arg5,N,N,arg1,arg1), DATA_TYPE POLYBENCH_2D(arg6,N,M,arg1,arg2))"},
    {"syr2k", "void syr2k(int arg1, int arg2, DATA_TYPE arg3, DATA_TYPE arg4, DATA_TYPE POLYBENCH_2D(arg5,N,N,arg1,arg1), DATA_TYPE POLYBENCH_2D(arg6,N,M,arg1,arg2), DATA_TYPE POLYBENCH_2D(arg7,N,M,arg1,arg2))"},
    {"trmm", "void trmm(int arg1, int arg2, DATA_TYPE arg3, DATA_TYPE POLYBENCH_2D(arg4,M,M,arg1,arg1), DATA_TYPE POLYBENCH_2D(arg5,M,N,arg1,arg2))"},
    {"Twomm", "void Twomm(int arg1, int arg2, int arg3, int arg4, DATA_TYPE arg5, DATA_TYPE arg6, DATA_TYPE POLYBENCH_2D(arg7,NI,NJ,arg1,arg2), DATA_TYPE POLYBENCH_2D(arg8,NI,NK,arg1,arg3), DATA_TYPE POLYBENCH_2D(arg9,NK,NJ,arg3,arg2), DATA_TYPE POLYBENCH_2D(arg10,NJ,NL,arg2,arg4), DATA_TYPE POLYBENCH_2D(arg11,NI,NL,arg1,arg4))"},
    {"Threemm", "void Threemm(int arg1, int arg2, int arg3, int arg4, int arg5, DATA_TYPE POLYBENCH_2D(arg6,NI,NJ,arg1,arg2), DATA_TYPE POLYBENCH_2D(arg7,NI,NK,arg1,arg3), DATA_TYPE POLYBENCH_2D(arg8,NK,NJ,arg3,arg2), DATA_TYPE POLYBENCH_2D(arg9,NJ,NL,arg2,arg4), DATA_TYPE POLYBENCH_2D(arg10,NJ,NM,arg2,arg5), DATA_TYPE POLYBENCH_2D(arg11,NM,NL,arg5,arg4), DATA_TYPE POLYBENCH_2D(arg12,NI,NL,arg1,arg4))"},
    {"symm", "void symm(int arg1, int arg2, DATA_TYPE arg3, DATA_TYPE arg4, DATA_TYPE POLYBENCH_2D(arg5,M,N,arg1,arg2), DATA_TYPE POLYBENCH_2D(arg6,M,M,arg1,arg1), DATA_TYPE POLYBENCH_2D(arg7,M,N,arg1,arg2))"},
    {"gesummv", "void gesummv(int arg1, DATA_TYPE arg2, DATA_TYPE arg3, DATA_TYPE POLYBENCH_2D(arg4,N,N,arg1,arg1), DATA_TYPE POLYBENCH_2D(arg5,N,N,arg1,arg1), DATA_TYPE POLYBENCH_1D(arg6,N,arg1), DATA_TYPE POLYBENCH_1D(arg7,N,arg1), DATA_TYPE POLYBENCH_1D(arg8,N,arg1))"},
    {"gemver", "void gemver(int arg1, DATA_TYPE arg2, DATA_TYPE arg3, DATA_TYPE POLYBENCH_2D(arg4,N,N,arg1,arg1), DATA_TYPE POLYBENCH_1D(arg5,N,arg1), DATA_TYPE POLYBENCH_1D(arg6,N,arg1), DATA_TYPE POLYBENCH_1D(arg7,N,arg1), DATA_TYPE POLYBENCH_1D(arg8,N,arg1), DATA_TYPE POLYBENCH_1D(arg9,N,arg1), DATA_TYPE POLYBENCH_1D(arg10,N,arg1), DATA_TYPE POLYBENCH_1D(arg11,N,arg1), DATA_TYPE POLYBENCH_1D(arg12,N,arg1))"},
    };


#endif