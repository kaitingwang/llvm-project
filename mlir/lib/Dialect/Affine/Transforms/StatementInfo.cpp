//===- ValidatorUtils.cpp ---- Utilities for validator related work -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous transformation utilities for the poly work
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/StatementInfo.h"
#include "mlir/Dialect/Affine/LabelChain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <string>

#define DEBUG_TYPE "statement-info"
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::presburger;

mlir::affine::StatementInfo::~StatementInfo() {}

void mlir::affine::StatementInfo::printDomain() {
  LLVM_DEBUG(llvm::dbgs() << "Domain:\n");
  for (int i = 0; i < (int)domain.size(); i++) {
    std::string t = "";
    for (long long num : domain[i]) {
      bool extraSpace = false;
      std::string s = std::to_string(num);
      if (t.size() && num < 0) {
        t.pop_back();
        extraSpace = true;
      }
      if (s.size() == 3) {
        s.push_back(' ');
        s.push_back(' ');
      }
      if (s.size() == 2) {
        s.push_back(' ');
        s.push_back(' ');
        s.push_back(' ');
      }
      if (s.size() == 1) {
        s.push_back(' ');
        s.push_back(' ');
        s.push_back(' ');
        s.push_back(' ');
      }
      if (extraSpace)
        s.push_back(' ');
      t = t + s;
    }
    LLVM_DEBUG(llvm::dbgs() << t << "\n");
  }
}

void mlir::affine::StatementInfo::printScattering(bool printCheckPoint) {
  SmallVector<SmallVector<int64_t>> &scatteringToPrint =
      printCheckPoint ? scatteringCheckPoint : scattering;

  if (printCheckPoint)
    LLVM_DEBUG(llvm::dbgs() << "scatteringCheckPoint:\n");
  else
    LLVM_DEBUG(llvm::dbgs() << "scattering:\n");
  for (int i = 0; i < (int)scatteringToPrint.size(); i++) {
    std::string t = "";
    for (long long num : scatteringToPrint[i]) {
      bool extraSpace = false;
      std::string s = std::to_string(num);
      if (num < 0 && t.size()) {
        extraSpace = true;
        t.pop_back();
      }
      if (s.size() == 3) {
        s.push_back(' ');
        s.push_back(' ');
      }
      if (s.size() == 2) {
        s.push_back(' ');
        s.push_back(' ');
        s.push_back(' ');
      }
      if (s.size() == 1) {
        s.push_back(' ');
        s.push_back(' ');
        s.push_back(' ');
        s.push_back(' ');
      }
      if (extraSpace)
        s.push_back(' ');
      t = t + s;
    }
    LLVM_DEBUG(llvm::dbgs() << t << "\n");
  }
}

void mlir::affine::StatementInfo::printConstantRow(bool printCheckPoint) {
  SmallVector<int> &constantRowsToPrint =
      printCheckPoint ? constantRowsCheckPoint : constantRows;
  if (printCheckPoint)
    LLVM_DEBUG(llvm::dbgs() << "constantRowsCheckPoint: ");
  else
    LLVM_DEBUG(llvm::dbgs() << "constantRows: ");
  for (int num : constantRowsToPrint) {
    LLVM_DEBUG(llvm::dbgs() << num << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

void mlir::affine::StatementInfo::LoopReorder(int x, int y) {
  LLVM_DEBUG(llvm::dbgs() << "Reorder " << x << " to " << y << "\n");
  if (scattering.size() == 0)
    return;

  transformationList.push_back(new LabelInterchange(x, y));
  for (int i = scattering.size() + 1; i < scattering[0].size(); i++) {
    std::swap(scattering[1 + 2 * x][i], scattering[1 + 2 * y][i]);
  }
}

void mlir::affine::StatementInfo::LoopSkew(int x, int y, int amount) {
  LLVM_DEBUG(llvm::dbgs() << "Skew " << x << " with " << y << "\n");
  if (scattering.size() == 0)
    return;

  transformationList.push_back(new LabelSkew(x, y, amount));
  for (int i = scattering.size() + 1; i < scattering[0].size(); i++) {
    scattering[1 + 2 * y][i] +=
        amount * scattering[1 + 2 * x][i];
    // 1 for eq/ineq
    // the row is always equal to 2 * maxRowCount + 1
    // after that we begin our reordering
  }
}

bool mlir::affine::StatementInfo::hasPrefixLabelChain(
    SmallVector<LabelBase *> labelChain) {
  SmallVector<LabelBase *> labelChain1;
  SmallVector<LabelBase *> labelChain2;
  for (auto *label : labelChain) {
    if (label->transformationType == Interchange ||
        label->transformationType == Skew)
      continue;
    labelChain2.push_back(label);
  }
  for (auto *label : transformationList) {
    if (label->transformationType == Interchange ||
        label->transformationType == Skew)
      continue;
    labelChain1.push_back(label);
  }
  if (labelChain2.size() == 0) {
    return true;
  }
  if (labelChain1.size() < labelChain2.size()) {
    return false;
  }
  // for (int i = 0; i < labelChain2.size(); i++) {
  //   if (*(labelChain2[i]) == *(labelChain1[i])) {
  //     continue;
  //   }
  //   return false;
  // }
  // return true;
  unsigned int j = 0;
  for (unsigned i = 0; i < labelChain1.size(); i++) {
    if (labelChain2.size() == j)
      return true;
    if (*(labelChain2[j]) == *(labelChain1[i])) {
      j++;
    }
  }
  return j == labelChain2.size();
}

void mlir::affine::StatementInfo::applySumTrick(int depth) {
  if (isFusedInto == 0 &&
      (isIteratorTiled.size() < 2 || isIteratorTiled[0] != isIteratorTiled[1]))
    return;
  if (scattering.size() < 4) {
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Apply sum trick" << "\n");
  for (int i = 0; i < (int)scattering[0].size() - 1; i++) {
    if (scattering[1][i] <= 0 && scattering[3][i] <= 0)
      scattering[1][i] += scattering[3][i];
    LLVM_DEBUG(llvm::dbgs() << scattering[1][i] << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

int mlir::affine::StatementInfo::isTiled() {
  for (LabelBase *label : transformationList)
    if (label->transformationType == Tile) {
      LabelTile *tileLabel = static_cast<LabelTile *>(label);
      return (tileLabel->tileSizes).size();
    }
  return 0;
}

bool mlir::affine::StatementInfo::isReordered() {
  for (LabelBase *label : transformationList)
    if (label->transformationType == Interchange) {
      return true;
    }
  return false;
}

void mlir::affine::StatementInfo::replayPostTilingPrimitives() {
  bool isAfterTile = false;
  int tileSize = 0;
  for (LabelBase *label : transformationList)
    if (label->transformationType == Tile) {
      isAfterTile = true;
      SmallVector<int64_t, 4> tileSizes =
          static_cast<LabelTile *>(label)->tileSizes;
      tileSize =
          tileSizes.size() - std::count(tileSizes.begin(), tileSizes.end(), 1);
    } else {
      if (isAfterTile) {
        if (label->transformationType == Interchange) {
          // redo reorder if it is targetting point loops
          LabelInterchange *reorder = static_cast<LabelInterchange *>(label);
          if (reorder->fromDepth >= tileSize && reorder->toDepth >= tileSize)
            LoopReorder(reorder->fromDepth, reorder->toDepth);
        } else if (label->transformationType == Skew) {
          // redo skew if it is targetting point loops. This operation is unsafe thus raise a warning
          llvm::errs() << "Warning: Skew after tile is unsafe!\n";
          LabelSkew *skew = static_cast<LabelSkew *>(label);
          if (skew->skewTargetDepth >= tileSize)
            LoopSkew(skew->skewTargetDepth, skew->skewByDepth, skew->amount);
        }
        // other transformation does not modify schedule, no need to redo
      }
    }
}

void mlir::affine::StatementInfo::applyUserSchedule() {
  assert(manualSetFlag && "Cannot apply user schedule without manual set.");

  int startPos = 2 * maxRowCount + 2;
  for (unsigned i = 0; i < userSchedule.size(); i++) {
    for (unsigned j = 0; j < userSchedule[i].size(); j++) {
      scattering[2 * i + 1][startPos + j] = -1 * userSchedule[i][j];
    }
  }
  // fill a from back to the front (necessary)
  // from bottom to top (not necessary)
  int row = userSchedule.size();
  for (int i = a.size() - 1; i >= 0; i--) {
    row--;
    int it = userSchedule[i].size();
    for (int j = a[i].size() - 1; j >= 0; j--) {
      it--;
      a[i][j] = userSchedule[row][it];
    }
  }
}