//===- TG.cpp ---- Transformation Graph related work -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Transformation Graph functions
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LabelChain.h"
#include "mlir/Dialect/Affine/PolyToolCorrectionUtils.h"
#include "mlir/Dialect/Affine/StatementInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>

#define DEBUG_TYPE "tg"
using llvm::dbgs;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::presburger;

void Node::addChild(Node *child) { children.push_back(child); }

Node::Node() {
  parents.clear();
  children.clear();
  isLeaf = false;
  isRoot = false;
}

Node::Node(int loopDepth, SmallVector<Node *> parents,
           SmallVector<Node *> children, AffineForOp loop, bool isLeaf,
           bool isRoot, LabelBase *transformation)
    : loopDepth{loopDepth}, parents{parents}, children{children}, loop{loop},
      isLeaf{isLeaf}, isRoot{isRoot}, transformation{transformation} {
  for (auto parent : parents)
    parent->addChild(this);
}

void Node::findAllNodes(Node *tgRoot) {
  (tgRoot)->AllNodes.insert(this);
  for (auto nei : children)
    nei->findAllNodes(tgRoot);
}

void Node::freeTree() {
  findAllNodes(this);
  LLVM_DEBUG(llvm::dbgs() << AllNodes.size() << "\n");
  SmallVector<Node *> toBeRemovedNodes;
  while (!AllNodes.empty()) {
    toBeRemovedNodes.push_back(*(AllNodes.begin()));
    AllNodes.erase(AllNodes.begin());
  }
  for (int i = 0; i < toBeRemovedNodes.size(); i++)
    delete toBeRemovedNodes[i];
}

SmallVector<Operation *> Node::findAffectedStatements(
    DenseMap<Operation *, affine::StatementInfo> &StatementInfoMap) {
  SmallVector<LabelBase *> labelChain;
  SmallVector<Operation *> ans;
  findAffectedStatementsHelper(StatementInfoMap, labelChain, ans);
  std::sort(ans.begin(), ans.end());
  auto last = std::unique(ans.begin(), ans.end());
  ans.erase(last, ans.end());
  std::sort(
      ans.begin(), ans.end(), [&StatementInfoMap](Operation *a, Operation *b) {
        return hasSmallerConstantRows(StatementInfoMap[a], StatementInfoMap[b]);
      });
  return ans;
}

void Node::findAffectedStatementsHelper(
    DenseMap<Operation *, affine::StatementInfo> &StatementInfoMap,
    SmallVector<LabelBase *> labelChain, SmallVector<Operation *> &ans) {
  if (!isRoot) {
    if (transformation->transformationType == FuseInto) {
      const LabelFuseInto *fuseIntoOp =
          static_cast<LabelFuseInto const *>(transformation);
      ans.push_back(fuseIntoOp->target);
      for (auto par : parents)
        par->findAffectedStatementsHelper(StatementInfoMap, labelChain, ans);
    } else {
      labelChain.push_back(transformation);
      for (auto par : parents)
        par->findAffectedStatementsHelper(StatementInfoMap, labelChain, ans);
      labelChain.pop_back();
    }
    return;
  }
  std::reverse(labelChain.begin(), labelChain.end());
  loop.walk([&](Operation *op) {
    if (StatementInfoMap.find(op) != StatementInfoMap.end() &&
        (StatementInfoMap[op].constantRows).size() > 0)
      if (StatementInfoMap[op].hasPrefixLabelChain(labelChain))
        ans.push_back(op);
  });
  std::reverse(labelChain.begin(), labelChain.end());
}
