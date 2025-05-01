//===- LabelChain.h - the Label Chain classes --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This header defines all the classes for label chains.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_LABELCHAIN_H
#define MLIR_DIALECT_AFFINE_LABELCHAIN_H

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include <unistd.h>
#include <utility>

namespace mlir {
namespace affine {
enum TransformationType {
  Match,
  Tile,
  Distribution,
  Interchange,
  Fusion,
  Parallel,
  Unroll,
  Skew,
  FuseInto,
  Vectorize,
  BlockReorder
};

struct LabelBase {
  TransformationType transformationType;
  virtual bool operator==(const LabelBase &rhs) = 0;
  virtual ~LabelBase() {};
};

struct LabelMatch : public LabelBase {
  LabelMatch() { transformationType = Match; }
  bool operator==(const LabelBase &rhs) override {
    return rhs.transformationType == Match;
  }
};

struct LabelTile : public LabelBase {
  SmallVector<int64_t, 4> tileSizes;
  SmallVector<int> prefixConstants;
  LabelTile(SmallVector<int64_t, 4> tileSizes, SmallVector<int> prefixConstants)
      : tileSizes(std::move(tileSizes)), prefixConstants(prefixConstants) {
    transformationType = Tile;
  }
  bool operator==(const LabelBase &rhs) override {
    if (rhs.transformationType != Tile)
      return false;
    const LabelTile *tileOp = static_cast<LabelTile const *>(&rhs);
    if ((tileOp->prefixConstants).size() < prefixConstants.size()) {
      return false;
    }
    for (int i = 0; i < (int)prefixConstants.size(); i++)
      if (prefixConstants[i] != (tileOp->prefixConstants)[i])
        return false;
    return true;
  }
  bool operator==(const LabelTile &rhs) {
    if ((rhs.prefixConstants).size() < prefixConstants.size()) {
      return false;
    }
    for (int i = 0; i < (int)prefixConstants.size(); i++)
      if (prefixConstants[i] != (rhs.prefixConstants)[i])
        return false;
    return true;
  }
};

struct LabelDistribution : public LabelBase {
  int setNumber;
  LabelDistribution(int setNumber) : setNumber(setNumber) {
    transformationType = Distribution;
  }
  bool operator==(const LabelBase &rhs) override {
    if (rhs.transformationType != Distribution)
      return false;
    const LabelDistribution *distributeOp =
        static_cast<LabelDistribution const *>(&rhs);
    return distributeOp->setNumber == this->setNumber;
  }
};

struct LabelInterchange : public LabelBase {
  int fromDepth;
  int toDepth;
  LabelInterchange(int fromDepth, int toDepth)
      : fromDepth(fromDepth), toDepth(toDepth) {
    transformationType = Interchange;
  }
  bool operator==(const LabelBase &rhs) override {
    if (rhs.transformationType != Interchange)
      return false;
    const LabelInterchange *reOrderOp =
        static_cast<LabelInterchange const *>(&rhs);
    return (reOrderOp->fromDepth == this->fromDepth) &&
           (reOrderOp->toDepth == this->toDepth);
  }
};

struct LabelSkew : public LabelBase {
  int skewTargetDepth;
  int skewByDepth;
  int amount;
  LabelSkew(int skewTargetDepth, int skewByDepth, int amount)
      : skewTargetDepth(skewTargetDepth), skewByDepth(skewByDepth),
        amount(amount) {
    transformationType = Skew;
  }
  bool operator==(const LabelBase &rhs) override {
    if (rhs.transformationType != Skew)
      return false;
    const LabelSkew *skewOp = static_cast<LabelSkew const *>(&rhs);
    return (skewOp->skewTargetDepth == this->skewTargetDepth) &&
           (skewOp->skewByDepth == this->skewByDepth) &&
           (amount == skewOp->amount);
  }
};

struct LabelFusion : public LabelBase {
  LabelFusion() { transformationType = Fusion; }
  bool operator==(const LabelBase &rhs) override {
    return rhs.transformationType == Fusion;
  }
};

struct LabelParallel : public LabelBase {
  SmallVector<int> affectedStatements;
  LabelParallel() { transformationType = Parallel; }
  LabelParallel(SmallVector<int> selectedOp) {
    affectedStatements = std::move(selectedOp);
    transformationType = Parallel;
  }
  bool operator==(const LabelBase &rhs) override {
    return rhs.transformationType == Parallel;
  }
};

struct LabelVectorize : public LabelBase {
  SmallVector<int> affectedStatements;
  LabelVectorize() { transformationType = Vectorize; }
  LabelVectorize(SmallVector<int> selectedOp) {
    affectedStatements = std::move(selectedOp);
    transformationType = Vectorize;
  }
  bool operator==(const LabelBase &rhs) override {
    return rhs.transformationType == Vectorize;
  }
};

struct LabelUnroll : public LabelBase {
  SmallVector<int> affectedStatements;
  int unrollFactor;
  LabelUnroll(int factor) {
    transformationType = Unroll;
    unrollFactor = factor;
  }
  LabelUnroll(SmallVector<int> selectedOp, int factor) {
    affectedStatements = std::move(selectedOp);
    unrollFactor = factor;
    transformationType = Unroll;
  }
  bool operator==(const LabelBase &rhs) override {
    if (rhs.transformationType != Unroll)
      return false;
    const LabelUnroll *unrollOp = static_cast<LabelUnroll const *>(&rhs);
    return unrollFactor == unrollOp->unrollFactor;
  }
};

struct LabelFuseInto : LabelBase {
  Operation *target;
  LabelFuseInto(Operation *target) : target{target} {
    transformationType = FuseInto;
  }
  bool operator==(const LabelBase &rhs) override {
    return rhs.transformationType == FuseInto;
  }
};

struct LabelBlockReorder : public LabelBase {
  int currPos;
  int nextPos;
  LabelBlockReorder(int scope, int curr, int next)
      : currPos(curr), nextPos(next) {
    transformationType = BlockReorder;
  }
  bool operator==(const LabelBase &rhs) override {
    if (rhs.transformationType != BlockReorder)
      return false;
    const LabelBlockReorder *reOrderOp =
        static_cast<LabelBlockReorder const *>(&rhs);
    return (reOrderOp->currPos == this->currPos) &&
           (reOrderOp->nextPos == this->nextPos);
  }
};
} // namespace affine
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_LABELCHAIN_H
