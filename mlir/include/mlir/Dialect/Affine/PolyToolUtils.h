//===- PolyToolUtils.h - Affine dialect utilities ---------------*- C++ -*-===//
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

#ifndef MLIR_DIALECT_AFFINE_POLYTOOLUTILS_H
#define MLIR_DIALECT_AFFINE_POLYTOOLUTILS_H

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/StatementInfo.h"
#include "mlir/IR/AffineMap.h"
#include <unistd.h>
#include <utility>

namespace mlir {
namespace affine {

int maxConstantInDepthi(
    int depth, SmallVector<Operation *> operationVector,
    DenseMap<Operation *, affine::StatementInfo> &StatementInfoMap);

// Tranforms an IntMatrix to a vector<vector<int>>
SmallVector<SmallVector<int64_t>>
IntMatrixToVector(const mlir::presburger::IntMatrix &m);

// Tranforms a vector<vector<int>> to an IntMatrix
mlir::presburger::IntMatrix
vectorToIntMatrix(const SmallVector<SmallVector<int64_t>> &vec);

// Gets a FlatLinearConstriant and returns it in the format of
// SmallVector<SmallVector<int64_t>>
SmallVector<SmallVector<int64_t>>
FlatConstraintToVector(FlatLinearConstraints flat);

// print out a FlatLinearConstraints with alignment
void printConstraints(FlatLinearConstraints &cnst, llvm::raw_ostream &os);

/// Extract int64_t values from the assumed ArrayAttr of IntegerAttr.
SmallVector<int64_t> extractFromI64ArrayAttr(Attribute attr);

SmallVector<std::pair<Value, int>>
getNumTileLoops(SmallVector<bool> &localTiledIVBitMap, Operation *targetOp,
                int numFuseIntoLayer = 0);

// Find the list of Affine Yeild and Affine For ops
// To transform constant row A to constant row B
// Without any extra information and by only use
// of constant rows
std::string reverseEngineerConstantRows(SmallVector<int> A, SmallVector<int> B,
                                        int loopDepth);

// count the dimension of a perfectly nested loop around the target op
size_t depthOfPerfectlyNestingAroundOp(Operation *op);

// Checks if a op is related to control flow, like if ops, for ops, yield ops,
// return ops, etc
bool isControlFlow(Operation *op);
} // namespace affine
} // namespace mlir

#endif