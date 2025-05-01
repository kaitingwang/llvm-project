//===- StatementInfo.h - StatementInfo class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This header defines all the classes for StatementInfo.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_STATEMENTINFO_H
#define MLIR_DIALECT_AFFINE_STATEMENTINFO_H

#include "LabelChain.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include <unistd.h>

namespace mlir {
namespace affine {
struct StatementInfo {
  int opId;
  int maxSymCount = 0;
  int maxTileSize = 0;
  int maxColCount = 0;
  int maxRowCount = 0;
  int localTileSize = 0;
  int isFusedInto = 0;
  int manualSetFlag = false;
  SmallVector<SmallVector<int64_t>> domain;
  SmallVector<SmallVector<int64_t>> scattering;
  SmallVector<SmallVector<int64_t>> userSchedule;
  SmallVector<SmallVector<int64_t>> scatteringCheckPoint;
  SmallVector<bool> isIteratorTiled;
  SmallVector<int> constantRows;
  SmallVector<int> constantRowsCheckPoint;
  SmallVector<SmallVector<int64_t>> a;
  SmallVector<SmallVector<int64_t>> b;
  SmallVector<SmallVector<int64_t>> c;

  // previously known as LabelChains
  // Keeps the transformations in the order they applied
  SmallVector<LabelBase *> transformationList;

  void printDomain();
  void printConstantRow(bool printCheckPoint = false);
  void printScattering(bool printCheckPoint = false);
  void applySumTrick(int depth);
  void applyUserSchedule();
  void LoopReorder(int x, int y);
  void LoopSkew(int x, int y, int amount);
  bool operator<(const StatementInfo &other) { return opId < other.opId; }
  bool operator==(const StatementInfo &other) { return opId == other.opId; }
  bool hasPrefixLabelChain(SmallVector<LabelBase *> labelChain);
  int isTiled();
  bool isReordered();
  void replayPostTilingPrimitives();
  SmallVector<int> &getConstantRows() { return constantRows; }
  ~StatementInfo();
};

} // namespace affine
} // namespace mlir

#endif
