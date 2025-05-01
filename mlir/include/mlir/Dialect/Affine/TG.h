//===- LabelChain.h - the TG classes --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines all the classes for Transformation Graph (TG).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_TG_H
#define MLIR_DIALECT_AFFINE_TG_H

#include "LabelChain.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include <set>
#include <unistd.h>

namespace mlir {
namespace affine {
struct StatementInfo;
struct Node {
  int loopDepth;
  SmallVector<Node *> parents;
  SmallVector<Node *> children;
  AffineForOp loop = nullptr; // this is necessary for the tgRoot
  bool isLeaf = false;
  bool isRoot = false;
  LabelBase *transformation = nullptr;
  Node(int loopDepth, SmallVector<Node *> parents, SmallVector<Node *> children,
       AffineForOp loop, bool isLeaf, bool isRoot, LabelBase *transformation);
  Node();
  ~Node() { delete transformation; }
  int getDepth() { return loopDepth; }
  void addChild(Node *child);
  void freeTree();
  SmallVector<Operation *> findAffectedStatements(
      DenseMap<Operation *, affine::StatementInfo> &StatementInfoMap);

private:
  std::set<Node *> AllNodes;
  void findAllNodes(Node *tgRoot);
  void findAffectedStatementsHelper(
      DenseMap<Operation *, affine::StatementInfo> &StatementInfoMap,
      SmallVector<LabelBase *> labelChain, SmallVector<Operation *> &ans);
};
} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_TG_H