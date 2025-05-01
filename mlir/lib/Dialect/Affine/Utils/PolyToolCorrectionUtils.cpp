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

#include "mlir/Dialect/Affine/PolyToolCorrectionUtils.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/StatementInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "correct-schedule"
using llvm::dbgs;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::presburger;

bool mlir::opInsideLoop(Operation *op, Operation *forOp) {
  Operation *currOp = op->getParentOp();
  while (currOp) {
    if (currOp == forOp)
      return true;
    currOp = currOp->getParentOp();
  }
  return false;
}

bool mlir::enclosedInALoop(Operation *op) {
  SmallVector<Value, 4> SV;
  getAffineIVs(*op, SV);
  return (SV.size() != 0);
}

void mlir::affine::printSchedules(
    DenseMap<Operation *, SmallVector<IntMatrix, 4>> &schedules,
    DenseMap<Operation *, SmallVector<int>> &constants,
    DenseMap<Operation *, int> &scheduleOrder,
    DenseMap<Operation *, int> &minTile, DenseMap<Operation *, int> &maxTile) {
  int order = 0;
  int totalPrinted = 0;
  while (totalPrinted < scheduleOrder.size()) {
    for (auto it = schedules.begin(); it != schedules.end(); it++) {
      Operation *op = it->first;
      if (scheduleOrder[op] == order) {
        LLVM_DEBUG(dbgs() << "-------------------------\n");
        LLVM_DEBUG(dbgs() << "scheduleOrder(" << scheduleOrder[op] << "): ";
                   op->dump());
        LLVM_DEBUG(dbgs() << "a:\n"; (it->second)[1].dump(); dbgs() << "\n";);
        LLVM_DEBUG(dbgs() << "b:\n"; (it->second)[0].dump(); dbgs() << "\n";);
        LLVM_DEBUG(dbgs() << "c:\n"; (it->second)[2].dump(); dbgs() << "\n";);
        LLVM_DEBUG(dbgs() << "constants: ");
        for (int num : constants[op])
          LLVM_DEBUG(dbgs() << num << " ");
        LLVM_DEBUG(dbgs() << "\n");
        LLVM_DEBUG(dbgs() << "min, max, tile settings: " << minTile[op] << " "
                          << maxTile[op] << "\n");
        totalPrinted++;
      }
    }
    order++;
  }
}

void mlir::affine::printOneDependenceEdge(dependenceEdge &e) {
  LLVM_DEBUG(dbgs() << "-------------------------\n");
  LLVM_DEBUG(e.src->dump(););
  LLVM_DEBUG(e.dst->dump(););
  LLVM_DEBUG(dbgs() << "depth: " << e.depth << "\n");
  LLVM_DEBUG(dbgs() << "parallelDepth: " << e.parallelDepth << "\n");
  LLVM_DEBUG(e.dependenceConstraints.dump());
  LLVM_DEBUG(dbgs() << "numSymbolVars: "
                    << e.dependenceConstraints.getNumSymbolVars() << "\n");
  LLVM_DEBUG(dbgs() << "numDimVars: " << e.dependenceConstraints.getNumDimVars()
                    << "\n");
  LLVM_DEBUG(dbgs() << "numLocalVars: "
                    << e.dependenceConstraints.getNumLocalVars() << "\n");
}
void mlir::affine::printDependenceEdges(SmallVector<dependenceEdge, 8> &deps) {
  unsigned rawCount = 0, warCount = 0, wawCount = 0;
  for (dependenceEdge e : deps) {
    printOneDependenceEdge(e);
    if (e.type == EdgeType::RAW)
      rawCount++;
    if (e.type == EdgeType::WAW)
      wawCount++;
    if (e.type == EdgeType::WAR)
      warCount++;
  }
  LLVM_DEBUG(dbgs() << "Total edges: " << deps.size() << " ");
  LLVM_DEBUG(dbgs() << "(raw:" << rawCount << ", war:" << warCount
                    << ", waw: " << wawCount << ")\n");
}

void mlir::affine::recordScheduleInfo(
    SmallVector<Operation *, 8> &statements,
    DenseMap<Operation *, SmallVector<int>> &constants,
    DenseMap<Operation *, int> &scheduleOrder, Operation *&prevOp,
    Operation *&currOp, int &stmtCount) {
  // getConstant
  SmallVector<int> result;
  int depth = mlir::affine::getNestingDepth(currOp);
  if (prevOp == nullptr) {
    result.push_back(0);
    for (int i = 0; i < depth; i++)
      result.push_back(0);
  } else {
    unsigned common = affine::getNumCommonSurroundingLoops(*currOp, *prevOp);
    for (unsigned int i = 0; i < common; i++)
      result.push_back(constants[prevOp][i]);
    result.push_back(constants[prevOp][common] + 1);
    for (unsigned int i = 0; i < depth - common; i++)
      result.push_back(0);
  }
  constants[currOp] = result;
  statements.push_back(currOp);
  if (prevOp == nullptr || prevOp->getParentOp() != currOp->getParentOp())
    stmtCount++;
  scheduleOrder[currOp] = stmtCount;
  prevOp = currOp;
}

void mlir::affine::addOptoMap(
    DenseMap<mlir::Value, SmallVector<Operation *>> &memoryMap, mlir::Value arg,
    Operation *op) {
  if (memoryMap.find(arg) != memoryMap.end()) {
    memoryMap[arg].push_back(op);
  } else {
    SmallVector<Operation *> newSV;
    newSV.push_back(op);
    memoryMap[arg] = newSV;
  }
}

void mlir::affine::recordDependenceEdge(
    SmallVector<dependenceEdge, 8> &dependenceEdges, Operation *writeOp,
    Operation *readOp, FlatAffineValueConstraints &dcs,
    DenseMap<mlir::Value, bool> &numSym, Value &arg, unsigned depth,
    EdgeType ty) {
  dependenceEdge e;
  e.src = writeOp;
  e.dst = readOp;
  e.dependenceConstraints = dcs;
  e.arg = arg;
  e.depth = depth;
  e.type = ty;
  e.isEmpty = false;
  e.parallelDepth = -1;
  dependenceEdges.push_back(e);
  SmallVector<Value, 4> aSymValues;
  dcs.getValues(dcs.getNumDimVars(), dcs.getNumDimAndSymbolVars(), &aSymValues);
  for (Value val : aSymValues) {
    if (numSym.find(val) == numSym.end())
      numSym[val] = true;
  }
}

void mlir::affine::computeFarkasLHS(SmallVector<IntMatrix, 4> srcSchedule,
                                    SmallVector<IntMatrix, 4> dstSchedule,
                                    IntMatrix &farkasLHS, Operation *src,
                                    Operation *dst,
                                    DenseMap<Operation *, int> &scheduleOrder,
                                    int maxDim, int maxSym, int numLocal,
                                    bool parallel) {
  LLVM_DEBUG(dbgs() << "farkasLHS maxDim:"
                    << " " << maxDim << " maxSym:" << maxSym
                    << " parallel:" << parallel << "\n");

  int multiplier = parallel ? -1 : 1;
  unsigned oldrow = farkasLHS.getNumRows();
  farkasLHS.insertRows(oldrow, 2 * maxDim + maxSym + numLocal + 1);
  int dstc = scheduleOrder[dst] * (maxDim + maxSym + 1);
  LLVM_DEBUG(dbgs() << "dst belongs to ScopStmt:" << scheduleOrder[dst]
                    << "\n");
  LLVM_DEBUG(dbgs() << "src belongs to ScopStmt:" << scheduleOrder[src]
                    << "\n");
  LLVM_DEBUG(dbgs() << "dstc:" << dstc << "\n");
  LLVM_DEBUG(dbgs() << "farkasLHS #rows:" << farkasLHS.getNumRows() << "\n");
  LLVM_DEBUG(dbgs() << "farkasLHS #cols:" << farkasLHS.getNumColumns() << "\n");
  // assume schedule matrix is square
  unsigned srcDepth = srcSchedule[0].getNumRows(),
           dstDepth = dstSchedule[0].getNumRows();
  LLVM_DEBUG(dbgs() << "srcDepth:" << srcDepth << "\n");
  LLVM_DEBUG(dbgs() << "dstDepth:" << dstDepth << "\n");

  // dst
  for (unsigned int i = 0; i < dstDepth; i++)
    for (unsigned int j = 0; j < dstDepth; j++)
      farkasLHS(i + maxDim + oldrow, dstc + j) =
          dstSchedule[0](j, i) * multiplier;
  for (int i = 0; i < maxSym; i++)
    farkasLHS(2 * maxDim + i + oldrow, dstc + maxDim + i) = 1 * multiplier;
  farkasLHS(2 * maxDim + maxSym + numLocal + oldrow, dstc + maxDim + maxSym) =
      1 * multiplier;
  // src
  int srcc = scheduleOrder[src] * (maxDim + maxSym + 1);
  for (unsigned int i = 0; i < srcDepth; i++)
    for (unsigned int j = 0; j < srcDepth; j++)
      farkasLHS(i + oldrow, j + srcc) = -1 * srcSchedule[0](j, i) * multiplier;
  for (int i = 0; i < maxSym; i++)
    farkasLHS(2 * maxDim + i + oldrow, srcc + maxDim + i) += -1 * multiplier;
  farkasLHS(2 * maxDim + maxSym + numLocal + oldrow, srcc + maxDim + maxSym) +=
      -1 * multiplier;
}

void mlir::affine::computeFarkasRHS(FlatAffineValueConstraints *d, IntMatrix &m,
                                    bool TRs,
                                    SmallVector<IntMatrix, 4> srcSchedule,
                                    SmallVector<IntMatrix, 4> dstSchedule,
                                    int maxDim, int maxSym, int numLocal) {

  unsigned numOfCols = d->getNumEqualities() * 2 + d->getNumInequalities() + 1;
  unsigned numOfRows = 2 * maxDim + maxSym + numLocal + 1; // Const

  unsigned oldrow = m.getNumRows();
  unsigned oldcol = m.getNumColumns();
  LLVM_DEBUG(dbgs() << "farkasRHS #rows:" << numOfRows + oldrow << "\n");
  LLVM_DEBUG(dbgs() << "farkasRHS #cols:" << numOfCols + oldcol << "\n");
  m.insertColumns(oldcol, numOfCols);
  m.insertRows(oldrow, numOfRows);
  m(m.getNumRows() - 1, oldcol) = 1;

  unsigned count = oldcol + 1;
  // Assume schedule matrix is square.
  unsigned srcDepth = srcSchedule[0].getNumRows(),
           dstDepth = dstSchedule[0].getNumRows();
  for (unsigned i = 0, e = d->getNumEqualities(); i < e; i++) {
    unsigned diff = 0;
    for (unsigned j = 0, f = d->getNumCols(); j < f; j++) {
      if (j == srcDepth)
        diff += maxDim - srcDepth;
      if (j == srcDepth + dstDepth)
        diff += maxDim - dstDepth;
      if (j == srcDepth + dstDepth + d->getNumSymbolVars())
        diff += maxSym - d->getNumSymbolVars();
      m(j + oldrow + diff, count) = d->atEq(i, j);
      m(j + oldrow + diff, count + 1) = -1 * (d->atEq(i, j));
    }
    count += 2;
  }
  for (unsigned i = 0, e = d->getNumInequalities(); i < e; ++i) {
    unsigned diff = 0;
    for (unsigned j = 0, f = d->getNumCols(); j < f; ++j) {
      if (j == srcDepth)
        diff += maxDim - srcDepth;
      if (j == srcDepth + dstDepth)
        diff += maxDim - dstDepth;
      if (j == srcDepth + dstDepth + d->getNumSymbolVars())
        diff += maxSym - d->getNumSymbolVars();
      m(j + oldrow + diff, count) = d->atIneq(i, j);
    }
    count++;
  }
}

// Parse an matrix object to an MLIR attribute.
Attribute mlir::affine::convertMatrixToArrayAttr(IntMatrix m,
                                                 OpBuilder &builder,
                                                 MLIRContext *ctx) {
  SmallVector<Attribute> out;
  for (unsigned i = 0; i < m.getNumRows(); i++) {
    SmallVector<Attribute> row;
    for (unsigned j = 0; j < m.getNumColumns(); j++) {
      row.push_back(IntegerAttr::get(builder.getI64Type(), int64_t(m(i, j))));
    }
    out.push_back(ArrayAttr::get(ctx, row));
  }
  return ArrayAttr::get(ctx, out);
}

// check if a SmallVector contains the target Value
bool mlir::affine::contains(SmallVector<Value> &arr, Value &target) {
  for (Value &v : arr) {
    if (v == target)
      return true;
  }
  return false;
}

// For the input operation, count the number of loops it's nested in.
// Also, collects all IV and symbols used in the loops
unsigned mlir::affine::getDepthAndLoopOperands(Operation *op,
                                               SmallVector<AffineForOp> &loops,
                                               SmallVector<Value> &symbols,
                                               OpBuilder &builder) {
  Operation *currOp = op;
  unsigned depth = 0;
  SmallVector<Value> inductionVars;
  while ((currOp = currOp->getParentOp())) {
    if (AffineForOp loop = dyn_cast<AffineForOp>(currOp)) {
      depth++;
      loops.push_back(loop);
      inductionVars.push_back(loop.getInductionVar());
    }
  }
  currOp = op;
  for (AffineForOp loop : loops) {
    for (Value operand : loop.getLowerBoundOperands()) {
      if (!isa<BlockArgument>(operand))
        if (arith::IndexCastOp iCast =
                dyn_cast<arith::IndexCastOp>(operand.getDefiningOp()))
          operand = iCast.getIn();
      if (!contains(inductionVars, operand) && !contains(symbols, operand))
        symbols.push_back(operand);
    }
    for (Value operand : loop.getUpperBoundOperands()) {
      if (!isa<BlockArgument>(operand))
        if (arith::IndexCastOp iCast =
                dyn_cast<arith::IndexCastOp>(operand.getDefiningOp()))
          operand = iCast.getIn();
      if (!contains(inductionVars, operand) && !contains(symbols, operand))
        symbols.push_back(operand);
    }
  }
  std::reverse(loops.begin(), loops.end());
  return depth;
}

// For the input operation, count the number of ifs it's nested in.
// Also, collects all IV and symbols used in the if
unsigned mlir::affine::getDepthAndIfOperands(Operation *op,
                                             SmallVector<AffineIfOp> &ifs,
                                             SmallVector<Value> &symbols,
                                             OpBuilder &builder) {
  Operation *currOp = op;
  unsigned depth = 0;
  while ((currOp = currOp->getParentOp())) {
    if (AffineIfOp conditional = dyn_cast<AffineIfOp>(currOp)) {
      depth++;
      ifs.push_back(conditional);
    }
  }
  std::reverse(ifs.begin(), ifs.end());
  return depth;
}

// Create a 2D vector in C++ from an MLIR ArrayAttr
SmallVector<SmallVector<int64_t>> mlir::affine::parseMatrix(ArrayAttr attr) {
  SmallVector<SmallVector<int64_t>> out;
  for (unsigned i = 0; i < attr.size(); i++) {
    ArrayAttr arr = attr[i].cast<ArrayAttr>();
    out.push_back(SmallVector<int64_t>(0));
    for (unsigned j = 0; j < arr.size(); j++) {
      out[i].push_back(arr[j].cast<IntegerAttr>().getInt());
    }
  }
  return out;
}

// Do a matmul with 2 2D c++ vectors
SmallVector<SmallVector<int64_t>>
mlir::affine::matmul(SmallVector<SmallVector<int64_t>> &a,
                     SmallVector<SmallVector<int64_t>> &b) {
  if (a.size() == 0)
    return a;
  SmallVector<SmallVector<int64_t>> out(a.size(),
                                        SmallVector<int64_t>(b[0].size(), 0));
  for (unsigned i = 0; i < a.size(); i++) {
    for (unsigned j = 0; j < b[0].size(); j++) {
      for (unsigned k = 0; k < a[0].size(); k++) {
        out[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return out;
}

//  Modified from getEnclosingAffineForAndIfOps from Util.cpp
void mlir::affine::getEnclosingAffineForOps(Operation &op,
                                            SmallVectorImpl<Operation *> *ops) {
  ops->clear();
  Operation *currOp = op.getParentOp();

  // Traverse up the hierarchy collecting all `affine.for`
  // operations.
  while (currOp) {
    if (isa<AffineForOp>(currOp))
      ops->push_back(currOp);
    currOp = currOp->getParentOp();
  }
  std::reverse(ops->begin(), ops->end());
}

void mlir::affine::injectCandidateSchedule(
    SmallVector<SmallVector<int64_t>> &combined,
    SmallVector<SmallVector<int64_t>> &c, SmallVector<int> &constants,
    affine::StatementInfo &info, int maxSym, int minTile, int maxTile) {
  // Re-calculate numTiledLoops as isIteratorTiled can be zero'ed out.
  int numTiledLoops = std::count(info.isIteratorTiled.begin(),
                                 info.isIteratorTiled.end(), true) +
                      info.isFusedInto;
  // Extract numRows,numCols to create the candidate schedule.
  int numRows = (info.scattering.size() - 1 - 2 * numTiledLoops) / 2;
  int numCols = info.scattering[0].size() - 1 - info.scattering.size() -
                maxSym - 1 - numTiledLoops;
  int startRow = 2 * numTiledLoops + 1;
  int startCol = info.scattering.size() + 1 + numTiledLoops;

  LLVM_DEBUG(llvm::dbgs() << "Schedule/Scattering Before injection:\n");
  LLVM_DEBUG(info.printConstantRow());
  LLVM_DEBUG(info.printScattering());

  // Rows get padded with zeros, thus schedules' NumRows is <= numRows.
  // For LU due to fuse_into, combined[0].size() <= numCols.
  assert(combined.size() <= numRows && combined[0].size() <= numCols &&
         "mismatch dimension");

  for (unsigned i = 0; i < combined.size(); i++) {
    for (unsigned j = 0; j < combined[0].size(); j++)
      info.scattering[2 * i + startRow][j + startCol] = -1 * combined[i][j];
    assert(numCols + startCol + c[i].size() == info.scattering[0].size() &&
           "mismatch columns");
    for (int j = 0; j < c[i].size(); j++)
      info.scattering[2 * i + startRow][j + startCol + numCols] = -1 * c[i][j];
  }
  LLVM_DEBUG(llvm::dbgs() << "Injected Final Schedule/Scattering:\n");
  LLVM_DEBUG(info.printConstantRow());
  LLVM_DEBUG(info.printScattering());
}

void mlir::affine::injectCandidateDomain(SmallVector<SmallVector<int64_t>> &a,
                                         affine::StatementInfo &info) {
  // Re-calculate numTiledLoops as isIteratorTiled can be zero'ed out.
  int numTiledLoops = std::count(info.isIteratorTiled.begin(),
                                 info.isIteratorTiled.end(), true) +
                      info.isFusedInto;

  // only applied to tiled domains
  if (numTiledLoops == 0 || a.size() == 0)
    return;

  int firstTiledLoopIdx = std::find(info.isIteratorTiled.begin(),
                                    info.isIteratorTiled.end(), true) -
                          info.isIteratorTiled.begin();

  LLVM_DEBUG(llvm::dbgs() << "Domain before injection:\n");
  LLVM_DEBUG(info.printDomain());
  LLVM_DEBUG(llvm::dbgs() << "The A matrix to be injected:\n");
  for (unsigned i = 0; i < a.size(); i++) {
    for (unsigned j = 0; j < a[i].size(); j++)
      LLVM_DEBUG(llvm::dbgs() << a[i][j] << " ";);
    LLVM_DEBUG(llvm::dbgs() << "\n";);
  }
  LLVM_DEBUG(info.printDomain());

  for (unsigned i = 0; i < a.size() - firstTiledLoopIdx; i++) {
    for (unsigned j = 0; j < a[0].size() - firstTiledLoopIdx; j++) {
      // Skip over the point loops, inject a into the tile loops
      // Each tile loop has 2 rows in the domain
      info.domain[2 * info.isIteratorTiled.size() + i * 2]
                 [1 + firstTiledLoopIdx + numTiledLoops + j] =
          a[i + firstTiledLoopIdx][j + firstTiledLoopIdx];
      info.domain[2 * info.isIteratorTiled.size() + i * 2 + 1]
                 [1 + firstTiledLoopIdx + numTiledLoops + j] =
          -1 * a[i + firstTiledLoopIdx][j + firstTiledLoopIdx];
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Injected Final domain:\n");
  LLVM_DEBUG(info.printDomain());
}

void mlir::affine::extractCandidateSchedule(
    DenseMap<Operation *, SmallVector<IntMatrix, 4>> &schedules,
    DenseMap<Operation *, SmallVector<int>> &constants,
    const DenseMap<Operation *, affine::StatementInfo> &StatementInfoMap,
    int maxSym, int maxDim, DenseMap<Operation *, int> &minTile,
    DenseMap<Operation *, int> &maxTile) {
  for (auto it = StatementInfoMap.begin(); it != StatementInfoMap.end(); it++) {
    affine::StatementInfo info = it->second;
    // Did not encounter TileOp, thus CheckPoint are still empty. Alternatively,
    // scattering and constantRows were empty in the first place.
    if (info.constantRowsCheckPoint.size() == 0)
      info.constantRowsCheckPoint = info.constantRows;
    if (info.scatteringCheckPoint.size() == 0)
      info.scatteringCheckPoint = info.scattering;
    info.printConstantRow(true);
    info.printScattering(true);
    if (info.scattering.size() < 1)
      continue;

    // Populate minTile/maxTile arrays.
    unsigned numTiledLoops = 0;
    unsigned trackLastTiledLoop = 0;
    for (int i = 0; i < info.isIteratorTiled.size(); i++) {
      if (info.isIteratorTiled[i]) {
        trackLastTiledLoop = i;
        numTiledLoops++;
      }
      if (numTiledLoops == 1) // Found minTile
        minTile[it->first] = trackLastTiledLoop;
    }
    maxTile[it->first] = trackLastTiledLoop + info.isFusedInto;
    if (numTiledLoops == 0) {
      minTile[it->first] = maxDim + 1;
      maxTile[it->first] = -1;
    }

    // Extract constantRows.
    constants[it->first] = info.constantRowsCheckPoint;

    // Now we obtained the `constants`, grow the 'a' and 'c' matrix.
    for (auto it = schedules.begin(); it != schedules.end(); it++) {
      Operation *op = it->first;
      (it->second)[1].insertRows((it->second)[1].getNumRows(),
                                 constants[op].size() -
                                     (it->second)[1].getNumRows() - 1);
      (it->second)[2].insertRows((it->second)[2].getNumRows(),
                                 constants[op].size() -
                                     (it->second)[2].getNumRows() - 1);
    }

    // Extract numRows,numCols to create the candidate schedule
    unsigned numRows = (info.scatteringCheckPoint.size() - 1) / 2;
    unsigned numCols = info.scatteringCheckPoint[0].size() - 1 -
                       info.scatteringCheckPoint.size() - maxSym - 1;
    unsigned startRow = 1;
    unsigned startCol = info.scatteringCheckPoint.size() + 1;

    // Rows get padded with zeros, thus schedules' NumRows is <= numRows
    // LU's fuse_into NumColumns <= numCols. This is because the stmts were
    // originally affected by 2 iterators (i.e. 2 columns) but after fuse_into,
    // it acquire another dimension.
    assert(schedules[it->first][0].getNumRows() <= numRows &&
           schedules[it->first][0].getNumColumns() <= numCols &&
           "mismatch dimension");

    for (unsigned i = 0; i < schedules[it->first][0].getNumRows(); i++)
      for (unsigned j = 0; j < schedules[it->first][0].getNumColumns(); j++)
        schedules[it->first][0](i, j) =
            std::abs(info.scatteringCheckPoint[2 * i + startRow][j + startCol]);
    LLVM_DEBUG(llvm::dbgs() << "Extracted Candidate Schedule:\n");
    LLVM_DEBUG(schedules[it->first][0].dump());
  }
}

void mlir::affine::computeDependencePolyhedron(
    SmallVector<dependenceEdge, 8> &dependenceEdges,
    DenseMap<mlir::Value, SmallVector<Operation *>> &readMap,
    DenseMap<mlir::Value, SmallVector<Operation *>> &writeMap,
    DenseMap<mlir::Value, bool> &memrefVals) {
  DenseMap<mlir::Value, bool> numSym;
  for (auto it = writeMap.begin(); it != writeMap.end(); it++) {
    mlir::Value arg = it->first;
    bool isSSA = (memrefVals.find(arg) == memrefVals.end());
    if (isSSA) // skip for now
      continue;
    for (Operation *writeOp : it->second) {
      if (!enclosedInALoop(writeOp))
        continue;
      MemRefAccess writeAccess(writeOp);
      if (readMap.find(arg) != readMap.end()) {
        for (Operation *readOp : readMap[arg]) {
          if (!enclosedInALoop(readOp))
            continue;
          unsigned commonLoops =
              affine::getNumCommonSurroundingLoops(*writeOp, *readOp);
          for (unsigned i = 1; i <= commonLoops + 1; i++) {
            FlatAffineValueConstraints readAfterWrite;
            MemRefAccess readAccess(readOp);
            DependenceResult result = checkMemrefAccessDependence(
                writeAccess, readAccess, i, &readAfterWrite, nullptr, false);

            if (hasDependence(result))
              recordDependenceEdge(dependenceEdges, writeOp, readOp,
                                   readAfterWrite, numSym, arg, i - 1,
                                   EdgeType::RAW);

            FlatAffineValueConstraints writeAfterRead;
            result = checkMemrefAccessDependence(
                readAccess, writeAccess, i, &writeAfterRead, nullptr, false);

            if (hasDependence(result))
              recordDependenceEdge(dependenceEdges, readOp, writeOp,
                                   writeAfterRead, numSym, arg, i - 1,
                                   EdgeType::WAR);
          }
        }
      }
      for (Operation *writeOp2 : it->second) {
        if (!enclosedInALoop(writeOp2))
          continue;
        MemRefAccess writeAccess2(writeOp2);
        unsigned commonLoops =
            affine::getNumCommonSurroundingLoops(*writeOp, *writeOp2);
        for (unsigned i = 1; i <= commonLoops + 1; i++) {
          FlatAffineValueConstraints writeAfterWrite;
          DependenceResult result = checkMemrefAccessDependence(
              writeAccess, writeAccess2, i, &writeAfterWrite, nullptr, false);
          if (hasDependence(result))
            recordDependenceEdge(dependenceEdges, writeOp, writeOp2,
                                 writeAfterWrite, numSym, arg, i - 1,
                                 EdgeType::WAW);
        }
      }
    }
  }
  printDependenceEdges(dependenceEdges);
}

DenseMap<Operation *, SmallVector<IntMatrix, 4>>
mlir::affine::initSchedule(SmallVector<Operation *, 8> &statements,
                           int numSym) {
  DenseMap<Operation *, SmallVector<IntMatrix, 4>> schedules;
  SmallVector<IntMatrix, 4> matrices;
  for (Operation *stmt : statements) {
    matrices.clear();
    int dim = mlir::affine::getNestingDepth(stmt);
    IntMatrix Tsc(dim, dim), Csc(dim, dim), ts(dim, numSym + 1);
    Tsc = Tsc.identity(dim);
    Csc = Csc.identity(dim);
    matrices.push_back(Tsc);
    matrices.push_back(Csc);
    matrices.push_back(ts);
    schedules[stmt] = matrices;
  }
  return schedules;
}

bool mlir::affine::allPolyEmpty(
    SmallVector<dependenceEdge, 8> &dependenceEdges) {
  for (dependenceEdge e : dependenceEdges) {
    if (!e.isEmpty && !(e.dependenceConstraints).isEmpty())
      return false;
    else if ((e.dependenceConstraints).isEmpty())
      e.isEmpty = true;
  }
  return true;
}

SmallVector<DynamicAPInt, 8> mlir::affine::correctIter(
    SmallVector<Operation *, 8> statements,
    DenseMap<Operation *, SmallVector<IntMatrix, 4>> &schedules,
    DenseMap<Operation *, SmallVector<int>> constants,
    SmallVector<dependenceEdge, 8> &dependenceEdges,
    DenseMap<Operation *, int> scheduleOrder, int curiter, int numStmt,
    int maxDim, int maxSym, int numLocal, DenseMap<Operation *, int> &minTile,
    DenseMap<Operation *, int> &maxTile,
    SmallVector<SmallVector<DynamicAPInt, 8>, 8> &integerSamples) {

  IntMatrix farkasRHS(0, 0);
  IntMatrix farkasLHS(0, numStmt * (maxDim + maxSym + 1));

  SmallVector<int64_t, 8> StrongsatisfyPositions;
  DenseMap<int, bool> ShouldUpdateOpOrder;
  for (dependenceEdge e : dependenceEdges) {
    numLocal += e.dependenceConstraints.getNumLocalVars();
    if (!e.isEmpty) {
      LLVM_DEBUG(e.src->dump());
      LLVM_DEBUG(e.dst->dump());
      ShouldUpdateOpOrder[scheduleOrder[e.src]] = true;
      ShouldUpdateOpOrder[scheduleOrder[e.dst]] = true;
      LLVM_DEBUG(e.dependenceConstraints.dump());
      LLVM_DEBUG(dbgs() << "TRs:" << e.TRs << "\n");
      computeFarkasRHS(&(e.dependenceConstraints), farkasRHS, e.TRs,
                       schedules[e.src], schedules[e.dst], maxDim, maxSym,
                       numLocal);
      LLVM_DEBUG(farkasRHS.dump());
      computeFarkasLHS(schedules[e.src], schedules[e.dst], farkasLHS, e.src,
                       e.dst, scheduleOrder, maxDim, maxSym, numLocal, false);
      LLVM_DEBUG(farkasLHS.dump());

      // Prevent access constants vector out of bound.
      int nextConstantDst = curiter + 1 < constants[e.dst].size()
                                ? constants[e.dst][curiter + 1]
                                : 0;
      int nextConstantSrc = curiter + 1 < constants[e.src].size()
                                ? constants[e.src][curiter + 1]
                                : 0;
      // Do not enforce strong satifaction in the last dimension
      // as this will get Jacobi to overskew.
      if ((curiter < minTile[e.src] || curiter > maxTile[e.src] ||
           curiter < minTile[e.dst] || curiter > maxTile[e.dst]
           //|| (maxDim - 1) == curiter
           ) &&
          (nextConstantSrc > nextConstantDst) &&
          scheduleOrder[e.src] != scheduleOrder[e.dst]) {
        StrongsatisfyPositions.push_back(farkasLHS.getNumRows() - 1);
        LLVM_DEBUG(dbgs() << "Strongly satisfied position found.\n");
      }
      if (e.parallelDepth == curiter) {
        computeFarkasRHS(&(e.dependenceConstraints), farkasRHS, e.TRs,
                         schedules[e.src], schedules[e.dst], maxDim, maxSym,
                         numLocal);
        computeFarkasLHS(schedules[e.src], schedules[e.dst], farkasLHS, e.src,
                         e.dst, scheduleOrder, maxDim, maxSym, numLocal, true);
        LLVM_DEBUG(farkasLHS.dump());
      }
    }
    numLocal -= e.dependenceConstraints.getNumLocalVars();
  }
  LLVM_DEBUG(dbgs() << "FarkasLHS and RHS done!\n");
  // LexMin prefers solution such as 001 over 100, prefers 001 over 002
  // Since we ensures LHS columns are added before RHS, we effectively
  // minimize solutions of LHS which are coefficients of the schedule we are
  // looking for.
  LexSimplex simplex(farkasLHS.getNumColumns() + farkasRHS.getNumColumns());

  IntMatrix simplexEq(farkasLHS.getNumRows(), farkasLHS.getNumColumns() +
                                                  farkasRHS.getNumColumns() +
                                                  1);
  // Adding LHS columns.
  for (int i = 0; i < farkasLHS.getNumRows(); i++) {
    for (int j = 0; j < farkasLHS.getNumColumns(); j++)
      simplexEq(i, j) = farkasLHS(i, j);
  }

  // Adding RHS columns.
  for (int i = 0; i < farkasRHS.getNumRows(); i++) {
    for (int j = 0; j < farkasRHS.getNumColumns(); j++)
      // -1 for LHS-RHS>=0
      simplexEq(i, j + farkasLHS.getNumColumns()) = -1 * farkasRHS(i, j);
  }
  for (int p : StrongsatisfyPositions)
    simplexEq(p, farkasLHS.getNumColumns() + farkasRHS.getNumColumns()) = -1;

  // Convert inequalities from constraints into equalities. This effectively
  // doubles the number of rows. Can you do this?
  for (int i = 0; i < simplexEq.getNumRows(); i++) {
    ArrayRef<DynamicAPInt> curRow = simplexEq.getRow(i);
    simplex.addEquality(curRow);
  }

  // Add ineqalities that each lambda is >=0.
  // Add ineqalities that each LHS vars >= 1 for the purpose of avoiding trivial
  // solution. i.e. LHS vars + (-1) >=0
  IntMatrix lowerbound(
      farkasLHS.getNumColumns() + farkasRHS.getNumColumns() + numStmt,
      farkasLHS.getNumColumns() + farkasRHS.getNumColumns() + 1);
  for (int i = 0; i < farkasLHS.getNumColumns(); i++) {
    lowerbound(i, i) = 1;
    lowerbound(i, farkasLHS.getNumColumns() + farkasRHS.getNumColumns()) = 0;
    simplex.addInequality(lowerbound.getRow(i));
  }

  for (int i = 0; i < farkasRHS.getNumColumns(); i++) {
    lowerbound(i + farkasLHS.getNumColumns(), i + farkasLHS.getNumColumns()) =
        1;
    simplex.addInequality(lowerbound.getRow(i + farkasLHS.getNumColumns()));
  }

  // Remove trivial solution.  That is, zeros for the coefficients for the
  // iterators (i.e. LHS vars) are not useful as it destroys full-rankness (i.e.
  // loosing axes) The easiest way to ensure Sum(all Ci) >= 0 is to impose that
  // a Ci must be > 0.
  for (int i = 0; i < numStmt; i++) {
    int j = curiter;
    lowerbound(farkasLHS.getNumColumns() + farkasRHS.getNumColumns() + i,
               j + i * (maxDim + maxSym + 1)) = 1;
    lowerbound(farkasLHS.getNumColumns() + farkasRHS.getNumColumns() + i,
               farkasLHS.getNumColumns() + farkasRHS.getNumColumns()) = -1;
    simplex.addInequality(lowerbound.getRow(farkasLHS.getNumColumns() +
                                            farkasRHS.getNumColumns() + i));
  }

  auto res = simplex.findIntegerLexMin();
  SmallVector<DynamicAPInt, 8> integerSample;
  if (!res.isEmpty()) {
    LLVM_DEBUG(dbgs() << "Found a solution! \n");
    for (auto elem : *res) {
      LLVM_DEBUG(elem.print(dbgs()));
      integerSample.push_back(elem);
      LLVM_DEBUG(dbgs() << " ");
    }
    LLVM_DEBUG(dbgs() << "\n");
  } else {
    LLVM_DEBUG(dbgs() << "No solution found! \n");
    return integerSample;
  }
  integerSamples.push_back(integerSample);

  int edgei = 0, lambdai = numStmt * (maxDim + maxSym +
                                      1); // TODO: number of statements = 2.
  while (edgei < dependenceEdges.size()) {
    dependenceEdge e = dependenceEdges[edgei];

    // The purpose of adding this flag is that grandschmit and partial tiling
    // testcase.  The point is that we check if the source and destination
    // are in different loops or not.  If they are in different loops
    // before tiling, the structure of partial tiling will make sure
    // that no dependency is between them and thus no correction needed.
    bool flag = false;
    // We guard this with a balanced inner loop condition.
    if (constants[e.dst].size() == constants[e.src].size()) {
      for (int i = 0; i < std::min(constants[e.dst].size() - 1,
                                   constants[e.src].size() - 1);
           i++) {
        if (constants[e.dst][i] != constants[e.src][i]) {
          flag = true;
          break;
        }
      }
    }

    if (e.parallelDepth == curiter && e.parallelDepth == e.depth) {
      dependenceEdges[edgei].isEmpty = true;
    } else if (!e.isEmpty &&
               (curiter < minTile[e.src] || curiter > maxTile[e.src] ||
                curiter < minTile[e.dst] || curiter > maxTile[e.dst] || flag)) {
      LLVM_DEBUG(
          dbgs() << "lamdai: " << lambdai << " eqnum: "
                 << (e.dependenceConstraints).getNumEqualities() << " ineqnum: "
                 << (e.dependenceConstraints).getNumInequalities() << "\n ");

      // Prevent access constants vector out of bound.
      int nextConstantDst = curiter + 1 < constants[e.dst].size()
                                ? constants[e.dst][curiter + 1]
                                : 0;
      int nextConstantSrc = curiter + 1 < constants[e.src].size()
                                ? constants[e.src][curiter + 1]
                                : 0;
      if (nextConstantSrc != nextConstantDst) {
        LLVM_DEBUG(
            dbgs() << "Remove edge due to strong satifaction condition 1.\n");
        lambdai += (e.dependenceConstraints).getNumEqualities() * 2 +
                   (e.dependenceConstraints).getNumInequalities() + 1;
        dependenceEdges[edgei].isEmpty = true;
      } else if (integerSample[lambdai] > 0) {
        LLVM_DEBUG(
            dbgs() << "Remove edge due to strong satifaction condition 2.\n");
        dependenceEdges[edgei].isEmpty = true;
        lambdai += (e.dependenceConstraints).getNumEqualities() * 2 +
                   (e.dependenceConstraints).getNumInequalities() + 1;
      } else {
        LLVM_DEBUG(
            dbgs() << "Update edge due to weak satifaction condition.\n");
        lambdai += (e.dependenceConstraints).getNumEqualities() * 2 + 1;
        for (int i = (e.dependenceConstraints).getNumInequalities() - 1; i >= 0;
             i--) {
          if (integerSample[lambdai + i] != 0 &&
              (curiter < minTile[e.src] || curiter > maxTile[e.src] ||
               curiter < minTile[e.dst] || curiter < minTile[e.dst])) {
            (dependenceEdges[edgei].dependenceConstraints)
                .addEquality((e.dependenceConstraints).getInequality(i));
          }
        }
        lambdai += (e.dependenceConstraints).getNumInequalities();
      }
    } else if (!e.isEmpty &&
               (curiter == maxTile[e.src] || curiter == maxTile[e.dst])) {
      for (int ti = minTile[e.src]; ti <= curiter; ti++) {
        int nextConstantDst =
            ti + 1 < constants[e.dst].size() ? constants[e.dst][ti + 1] : 0;
        int nextConstantSrc =
            ti + 1 < constants[e.src].size() ? constants[e.src][ti + 1] : 0;
        if (nextConstantDst != nextConstantSrc ||
            integerSamples[ti][lambdai] > 0) {
          dependenceEdges[edgei].isEmpty = true;
          break;
        }
      }
      if (dependenceEdges[edgei].isEmpty) {
        lambdai += (e.dependenceConstraints).getNumEqualities() * 2 +
                   (e.dependenceConstraints).getNumInequalities() + 1;
      } else {
        lambdai += (e.dependenceConstraints).getNumEqualities() * 2 + 1;

        int ub = (e.dependenceConstraints).getNumInequalities() - 1;
        for (int i = ub; i >= 0; i--) {
          int cur_sum = 0;
          for (int ti = minTile[e.src]; ti <= curiter; ti++) {
            cur_sum += int64_t(integerSamples[ti][lambdai + i]);
          }
          if (cur_sum == 0)
            (dependenceEdges[edgei].dependenceConstraints)
                .addEquality((e.dependenceConstraints).getInequality(i));
        }

        lambdai += (e.dependenceConstraints).getNumInequalities();
      }
    }
    edgei++;
  }
  return integerSample;
}
