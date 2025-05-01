//===- PolyToolCorrectionUtil.h - Affine dialect utilities ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares a set of utilities for the validator passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_VALIDATORUTILS_H
#define MLIR_DIALECT_AFFINE_VALIDATORUTILS_H

#include "TG.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineMap.h"
#include <unistd.h>

namespace mlir {
bool enclosedInALoop(Operation *op);
bool opInsideLoop(Operation *op, Operation *forOp);

namespace affine {

bool hasSmallerConstantRows(const StatementInfo &s1, const StatementInfo &s2);
bool hasSmallerConstantRows(const SmallVector<int> s1, const StatementInfo &s2);

enum class EdgeType { RAW, WAR, WAW };

struct dependenceEdge {
  Value arg;
  bool TRs;
  unsigned depth;
  Operation *src;
  Operation *dst;
  EdgeType type;
  bool isEmpty;
  // depth starts from 0
  int parallelDepth;
  FlatAffineValueConstraints dependenceConstraints;
};

Attribute convertMatrixToArrayAttr(mlir::presburger::IntMatrix m,
                                   OpBuilder &builder, MLIRContext *ctx);
void printOneDependenceEdge(dependenceEdge &e);
void printDependenceEdges(SmallVector<dependenceEdge, 8> &deps);
void printSchedules(
    DenseMap<Operation *, SmallVector<mlir::presburger::IntMatrix, 4>>
        &schedules,
    DenseMap<Operation *, SmallVector<int>> &constants,
    DenseMap<Operation *, int> &scheduleOrder,
    DenseMap<Operation *, int> &minTile, DenseMap<Operation *, int> &maxTile);
void recordDependenceEdge(SmallVector<dependenceEdge, 8> &dependenceEdges,
                          Operation *writeOp, Operation *readOp,
                          FlatAffineValueConstraints &dcs,
                          DenseMap<mlir::Value, bool> &numSym, Value &arg,
                          unsigned depth, EdgeType ty);
void addOptoMap(DenseMap<mlir::Value, SmallVector<Operation *>> &memoryMap,
                mlir::Value arg, Operation *op);
void recordScheduleInfo(SmallVector<Operation *, 8> &statements,
                        DenseMap<Operation *, SmallVector<int>> &constants,
                        DenseMap<Operation *, int> &scheduleOrder,
                        Operation *&prevOp, Operation *&currOp, int &stmtCount);
void computeFarkasLHS(SmallVector<mlir::presburger::IntMatrix, 4> srcSchedule,
                      SmallVector<mlir::presburger::IntMatrix, 4> dstSchedule,
                      mlir::presburger::IntMatrix &farkasLHS, Operation *src,
                      Operation *dst, DenseMap<Operation *, int> &scheduleOrder,
                      int maxDim, int maxSym, int numLocal, bool parallel);
void computeFarkasRHS(FlatAffineValueConstraints *d,
                      mlir::presburger::IntMatrix &m, bool TRs,
                      SmallVector<mlir::presburger::IntMatrix, 4> srcSchedule,
                      SmallVector<mlir::presburger::IntMatrix, 4> dstSchedule,
                      int maxDim, int maxSym, int numLocal);

// check if a SmallVector contains the target Value
bool contains(SmallVector<Value> &arr, Value &target);

// For the input operation, count the number of loops it's nested in.
// Also, collects all IV and symbols used in the loops
unsigned getDepthAndLoopOperands(Operation *op, SmallVector<AffineForOp> &loops,
                                 SmallVector<Value> &symbols,
                                 OpBuilder &builder);

unsigned getDepthAndIfOperands(Operation *op, SmallVector<AffineIfOp> &ifs,
                               SmallVector<Value> &symbols, OpBuilder &builder);

SmallVector<SmallVector<int64_t>> parseMatrix(ArrayAttr attr);

// Do a matmul with 2 2D c++ vectors
SmallVector<SmallVector<int64_t>> matmul(SmallVector<SmallVector<int64_t>> &a,
                                         SmallVector<SmallVector<int64_t>> &b);

//  Modified from getEnclosingAffineForAndIfOps from Util.cpp
void getEnclosingAffineForOps(Operation &op, SmallVectorImpl<Operation *> *ops);

void injectCandidateSchedule(SmallVector<SmallVector<int64_t>> &combined,
                             SmallVector<SmallVector<int64_t>> &c,
                             SmallVector<int> &constants,
                             affine::StatementInfo &info, int maxSym,
                             int minTile, int maxTile);

void injectCandidateDomain(SmallVector<SmallVector<int64_t>> &a,
                           affine::StatementInfo &info);

void extractCandidateSchedule(
    DenseMap<Operation *, SmallVector<mlir::presburger::IntMatrix, 4>>
        &schedules,
    DenseMap<Operation *, SmallVector<int>> &constants,
    const DenseMap<Operation *, affine::StatementInfo> &StatementInfoMap,
    int maxSym, int maxDim, DenseMap<Operation *, int> &minTile,
    DenseMap<Operation *, int> &maxTile);

void computeDependencePolyhedron(
    SmallVector<dependenceEdge, 8> &dependenceEdges,
    DenseMap<mlir::Value, SmallVector<Operation *>> &readMap,
    DenseMap<mlir::Value, SmallVector<Operation *>> &writeMap,
    DenseMap<mlir::Value, bool> &memrefVals);

DenseMap<Operation *, SmallVector<mlir::presburger::IntMatrix, 4>>
initSchedule(SmallVector<Operation *, 8> &statements, int numSym);

bool allPolyEmpty(SmallVector<dependenceEdge, 8> &dependenceEdges);

SmallVector<DynamicAPInt, 8>
correctIter(SmallVector<Operation *, 8> statements,
            DenseMap<Operation *, SmallVector<mlir::presburger::IntMatrix, 4>>
                &schedules,
            DenseMap<Operation *, SmallVector<int>> constants,
            SmallVector<dependenceEdge, 8> &dependenceEdges,
            DenseMap<Operation *, int> scheduleOrder, int curiter, int numStmt,
            int maxDim, int maxSym, int numLocal,
            DenseMap<Operation *, int> &minTile,
            DenseMap<Operation *, int> &maxTile,
            SmallVector<SmallVector<DynamicAPInt, 8>, 8> &integerSamples);

} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_VALIDATORUTILS_H
