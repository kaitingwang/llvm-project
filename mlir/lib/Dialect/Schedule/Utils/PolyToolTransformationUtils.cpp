//===- PolyTool.cpp --- New Design of the Entire Tool ------ -------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Schedule/PolyToolTransformationUtils.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>

using namespace mlir;

#define DEBUG_TYPE "polytool-array-expansion"

SmallVector<Value>
getOperandListForNewMap(OperandRange oldOperands, unsigned numExistingDim,
                        unsigned numTotalDim, SmallVector<Value> &ivsAroundOp,
                        const SmallVector<int64_t> &newSizes) {
  SmallVector<Value> newOperands;

  assert(oldOperands.size() >= numExistingDim &&
         "oldOperands should contain the dim and symbol operands of the "
         "original map.");

  // put the existing dimension operands at the front, so that the position of
  // them in the affine expr does not need to be changed
  newOperands.insert(newOperands.begin(), oldOperands.begin(),
                     oldOperands.begin() + numExistingDim);

  // append the new dimensions
  // start from outer most dimension IV, and avoid duplicates
  size_t i = 0;
  while (newOperands.size() < numTotalDim) {
    if (newSizes[i] != 1) {
      if (i >= ivsAroundOp.size()) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "Warning: Not enough loop dimensions to expand the array.\n");
        newOperands.push_back(ivsAroundOp[ivsAroundOp.size() - 1]);
      } else
        newOperands.push_back(ivsAroundOp[i]);
    }
    i++;
  }

  // append the original symbols at the end
  newOperands.insert(newOperands.end(), oldOperands.begin() + numExistingDim,
                     oldOperands.end());

  return newOperands;
}

AffineMap getAffineMapForNewAllocaMemRef(AffineMap oldMap,
                                         MemRefType newMemRefTy,
                                         bool isSingleValueAlloca) {
  SmallVector<AffineExpr> newMapResults;

  // number of new dimension variable
  int numNewDim = 0;
  for (int64_t i = 0;
       i < newMemRefTy.getRank() -
               (isSingleValueAlloca ? 0 : oldMap.getNumResults());
       i++) {
    AffineExpr newResultAtI;
    if (newMemRefTy.getDimSize(i) == 1)
      newResultAtI = getAffineConstantExpr(0, oldMap.getContext());
    else {
      newResultAtI = getAffineDimExpr(numNewDim + oldMap.getNumDims(),
                                      oldMap.getContext());
      numNewDim++;
    }
    newMapResults.push_back(newResultAtI);
  }
  // append the original map results at the end of the current map results if
  // needed
  if (!isSingleValueAlloca)
    newMapResults.insert(newMapResults.end(), oldMap.getResults().begin(),
                         oldMap.getResults().end());

  return AffineMap::get(oldMap.getNumDims() + numNewDim, oldMap.getNumSymbols(),
                        newMapResults, oldMap.getContext());
}

memref::AllocaOp
mlir::schedule::expandAlloca(memref::AllocaOp alloca,
                             const SmallVector<int64_t> &sizes) {
  OpBuilder b(alloca->getContext());
  b.setInsertionPoint(alloca);

  SmallVector<int64_t> newAllocaSizes;
  // add in the expanded size before the original sizes
  for (int64_t newSizes : sizes)
    newAllocaSizes.push_back(newSizes);

  ArrayRef<int64_t> origSizes = alloca.getMemref().getType().getShape();
  bool isSingleValueAlloca = alloca.getMemref().getType().getNumElements() <= 1;

  // no need to care about original size if it is just a single value
  if (!isSingleValueAlloca)
    for (int64_t origSize : origSizes)
      newAllocaSizes.push_back(origSize);

  MemRefType newTy = MemRefType::get(
      newAllocaSizes, alloca.getMemref().getType().getElementType());
  memref::AllocaOp newOp = b.create<memref::AllocaOp>(alloca.getLoc(), newTy);

  SmallVector<Operation *> toDelete; // prevent concurrent modification
  SmallVector<Operation *> opUsingAlloca;
  for (OpOperand &use : alloca.getMemref().getUses())
    opUsingAlloca.push_back(use.getOwner());
  // opUsingAlloca now contains all op using this alloca from top to bottom
  std::reverse(opUsingAlloca.begin(), opUsingAlloca.end());

  for (OpOperand &use : alloca.getMemref().getUses()) {
    if (affine::AffineLoadOp load =
            dyn_cast<affine::AffineLoadOp>(use.getOwner())) {
      AffineMap newMap = getAffineMapForNewAllocaMemRef(load.getMap(), newTy,
                                                        isSingleValueAlloca);

      // get the affine loop IVs around the alloca
      SmallVector<Value> ivs;
      affine::getAffineIVs(*load.getOperation(), ivs);
      SmallVector<Value> newOperands =
          getOperandListForNewMap(load.getIndices(), load.getMap().getNumDims(),
                                  newMap.getNumDims(), ivs, sizes);

      b.setInsertionPoint(load);
      affine::AffineLoadOp newLoad = b.create<affine::AffineLoadOp>(
          load->getLoc(), newOp, newMap, newOperands);
      load.getResult().replaceAllUsesWith(newLoad.getResult());
      newLoad->setAttrs(load->getRawDictionaryAttrs());

      toDelete.push_back(load.getOperation());
    } else if (affine::AffineStoreOp store =
                   dyn_cast<affine::AffineStoreOp>(use.getOwner())) {
      AffineMap newMap = getAffineMapForNewAllocaMemRef(store.getMap(), newTy,
                                                        isSingleValueAlloca);

      // get the affine loop IVs around the alloca
      SmallVector<Value> ivs;
      affine::getAffineIVs(*store.getOperation(), ivs);
      // change the order to be inner most to outer most
      SmallVector<Value> newOperands = getOperandListForNewMap(
          store.getIndices(), store.getMap().getNumDims(), newMap.getNumDims(),
          ivs, sizes);

      b.setInsertionPoint(store);
      affine::AffineStoreOp newStore = b.create<affine::AffineStoreOp>(
          store->getLoc(), store.getValue(), newOp, newMap, newOperands);
      newStore->setAttrs(store->getRawDictionaryAttrs());

      toDelete.push_back(store.getOperation());
    } else
      LLVM_DEBUG(llvm::dbgs() << "Warning: Unrecognized user of alloca op.\n");
  }

  for (Operation *op : toDelete)
    op->erase();

  alloca->erase();
  return newOp;
}
