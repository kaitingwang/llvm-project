#include "mlir/Dialect/Affine/PolyToolUtils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/PolyToolCorrectionUtils.h"
#include "mlir/Dialect/Affine/StatementInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cloog/cloog.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <stdio.h>
#include <string>
#include <utility>

using namespace std;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::presburger;

#define DEBUG_TYPE "polytool"

// Convert an IntMatrix to a vector<vector<int>>.
SmallVector<SmallVector<int64_t>>
mlir::affine::IntMatrixToVector(const mlir::presburger::IntMatrix &m) {
  SmallVector<SmallVector<int64_t>> out;
  for (unsigned i = 0; i < m.getNumRows(); i++) {
    SmallVector<int64_t> row;
    for (unsigned j = 0; j < m.getNumColumns(); j++) {
      row.push_back(int64_t(m(i, j)));
    }
    out.push_back(row);
  }
  return out;
}

// Tranforms a vector<vector<int>> to an IntMatrix
mlir::presburger::IntMatrix
vectorToIntMatrix(const SmallVector<SmallVector<int64_t>> &vec) {
  if (vec.size() == 0 || vec[0].size() == 0)
    return mlir::presburger::IntMatrix(0, 0);

  mlir::presburger::IntMatrix out(vec.size(), vec[0].size());
  for (unsigned i = 0; i < vec.size(); i++)
    for (unsigned j = 0; j < vec[0].size(); j++)
      out(i, j) = vec[i][j];

  return out;
}

// Gets a FlatLinearConstriant and returns it in the format of.
// SmallVector<SmallVector<int64_t>>.
SmallVector<SmallVector<int64_t>>
mlir::affine::FlatConstraintToVector(FlatLinearConstraints flat) {
  SmallVector<SmallVector<int64_t>> ans;
  for (unsigned i = 0, e = flat.getNumEqualities(); i < e; ++i) {
    SmallVector<int64_t> tmp;
    tmp.push_back(0); // for equality
    for (unsigned j = 0, f = flat.getNumCols(); j < f; ++j) {
      tmp.push_back(flat.atEq64(i, j));
    }
    ans.push_back(tmp);
  }

  for (unsigned i = 0, e = flat.getNumInequalities(); i < e; ++i) {
    SmallVector<int64_t> tmp;
    tmp.push_back(1);
    for (unsigned j = 0, f = flat.getNumCols(); j < f; ++j) {
      tmp.push_back(flat.atIneq64(i, j));
    }
    ans.push_back(tmp);
  }
  return ans;
}

bool mlir::affine::hasSmallerConstantRows(const StatementInfo &s1,
                                          const StatementInfo &s2) {
  for (unsigned i = 0; i < (s1.constantRows).size(); i++) {
    if ((s2.constantRows).size() <= i) {
      return false;
    }
    if ((s1.constantRows)[i] < (s2.constantRows)[i]) {
      return true;
    }
    if ((s2.constantRows)[i] < (s1.constantRows)[i]) {
      return false;
    }
  }
  return false;
}

bool mlir::affine::hasSmallerConstantRows(const SmallVector<int> s1,
                                          const StatementInfo &s2) {
  for (int i = 0; i < s1.size(); i++) {
    if ((int)((s2.constantRows).size()) <= i)
      return false;
    if (s1[i] < (s2.constantRows)[i])
      return true;
    if ((s2.constantRows)[i] < s1[i])
      return false;
  }
  return false;
}

std::string mlir::affine::reverseEngineerConstantRows(SmallVector<int> A,
                                                      SmallVector<int> B,
                                                      int loopDepth) {
  // The technique is known as state-based BFS in competitive programming
  // Our initial state if A and our target state is B and we would
  // like to find the shortest path to transform A to B
  // the code is very similar to the standard and classic BFS
  std::map<SmallVector<int>, std::string> d;
  std::map<SmallVector<int>, int> outputCounterMap;
  std::map<SmallVector<int>, int> currentIteratorMap;
  std::map<SmallVector<int>, int> iteratorDomainMap;
  std::map<SmallVector<int>, int> currentDepthMap;
  std::queue<SmallVector<int>> q;
  q.push(A);
  d[A] = "";
  outputCounterMap[A] = 1;
  currentIteratorMap[A] = 0;
  iteratorDomainMap[A] = 0;
  currentDepthMap[A] = loopDepth + 1;
  LLVM_DEBUG(llvm::dbgs() << "Reverse Engineer wants to find sequence from: ");
  for (auto nei : A) {
    LLVM_DEBUG(llvm::dbgs() << nei << ", ");
  }
  LLVM_DEBUG(llvm::dbgs() << " To: ");
  for (auto nei : B) {
    LLVM_DEBUG(llvm::dbgs() << nei << ", ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
  while (q.size()) {
    SmallVector<int> x = q.front();
    q.pop();
    SmallVector<int> original = x;
    std::string distance = d[x];
    if (x == B) {
      LLVM_DEBUG(llvm::dbgs() << "Solution:\n");
      for (auto nei : distance) {
        LLVM_DEBUG(llvm::dbgs() << nei << " ");
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
      return distance;
    }
    int outputCounter = outputCounterMap[x];
    int currentIterator = currentIteratorMap[x];
    int currentDepth = currentDepthMap[x];
    int iteratorDomain = iteratorDomainMap[x];

    // Add a single statement
    distance.push_back('S');
    int tmp = -1;
    if (x.size()) {
      tmp = x.back();
      x.pop_back();
    }
    tmp++;
    x.push_back(tmp);
    if (d.find(x) == d.end()) {
      q.push(x);
      d[x] = distance;
      outputCounterMap[x] = outputCounter;
      currentIteratorMap[x] = currentIterator;
      currentDepthMap[x] = currentDepth;
      iteratorDomainMap[x] = iteratorDomain;
    }

    // Add a single For Op
    x = original;
    distance = d[x];
    outputCounter = outputCounterMap[x];
    currentIterator = currentIteratorMap[x];
    currentDepth = currentDepthMap[x];
    iteratorDomain = iteratorDomainMap[x];
    distance.push_back('F');
    x.push_back(0);
    currentDepth++;
    outputCounter++;
    if (d.find(x) == d.end()) {
      q.push(x);
      d[x] = distance;
      outputCounterMap[x] = outputCounter;
      currentIteratorMap[x] = currentIterator;
      currentDepthMap[x] = currentDepth;
      iteratorDomainMap[x] = iteratorDomain;
    }

    // Add a single Yeild Op
    x = original;
    if (x.size() >= 2) {
      distance = d[x];
      outputCounter = outputCounterMap[x];
      currentIterator = currentIteratorMap[x];
      currentDepth = currentDepthMap[x];
      iteratorDomain = iteratorDomainMap[x];
      distance.push_back('Y');
      currentIterator--;
      x.pop_back();
      iteratorDomain = 0;
      int temp = x.back() + 1;
      x.pop_back();
      x.push_back(temp);
      if (d.find(x) == d.end()) {
        q.push(x);
        d[x] = distance;
        outputCounterMap[x] = outputCounter;
        currentIteratorMap[x] = currentIterator;
        currentDepthMap[x] = currentDepth;
        iteratorDomainMap[x] = iteratorDomain;
      }
    }
  }
}

void mlir::affine::printConstraints(FlatLinearConstraints &cnst,
                                    llvm::raw_ostream &os) {
  os << cnst.getNumInequalities() + cnst.getNumEqualities() << " "
     << (cnst.getNumCols() + 1) << "\n";

  constexpr unsigned MIN_SPACING = 1;
  PrintTableMetrics ptm = {
      0, 0, "-"};

  for (unsigned row = 0; row < cnst.getNumEqualities(); ++row) {
    for (unsigned column = 0; column < cnst.getNumCols(); ++column) {
      updatePrintMetrics<DynamicAPInt>(cnst.atEq(row, column), ptm);
    }
  }

  for (unsigned i = 0, e = cnst.getNumEqualities(); i < e; ++i) {
    os << "0 "; // 0 for equality
    for (unsigned j = 0, f = cnst.getNumCols(); j < f; ++j) {
      // os << cnst.atEq(i, j) << " ";
      printWithPrintMetrics<DynamicAPInt>(os, cnst.atEq(i, j), MIN_SPACING,
                                          ptm);
    }
    os << "\n";
  }

  for (unsigned row = 0; row < cnst.getNumInequalities(); ++row) {
    for (unsigned column = 0; column < cnst.getNumCols(); ++column) {
      updatePrintMetrics<DynamicAPInt>(cnst.atIneq(row, column), ptm);
    }
  }

  for (unsigned i = 0, e = cnst.getNumInequalities(); i < e; ++i) {
    os << "1 "; // 0 for equality
    for (unsigned j = 0, f = cnst.getNumCols(); j < f; ++j) {
      // os << cnst.atIneq(i, j) << " ";
      printWithPrintMetrics<DynamicAPInt>(os, cnst.atIneq(i, j), MIN_SPACING,
                                          ptm);
    }
    os << "\n";
  }
}

SmallVector<std::pair<Value, int>>
mlir::affine::getNumTileLoops(SmallVector<bool> &localTiledIVBitMap,
                              Operation *targetOp, int numFuseIntoLayer) {
  SmallVector<std::pair<Value, int>> surroundingLoopIVs;
  SmallVector<Value> ivs;
  getAffineIVs(*targetOp, ivs);
  int numTileLoops = 0;
  for (unsigned i = 0; i < ivs.size(); i++) {
    unsigned cur = i;
    while (cur < localTiledIVBitMap.size() && localTiledIVBitMap[cur]) {
      localTiledIVBitMap[cur] = false;
      cur++;
      numTileLoops++;
    }
    if (numTileLoops > 0) {
      numTileLoops += numFuseIntoLayer;
      numFuseIntoLayer = 0;
    }
    surroundingLoopIVs.push_back(std::make_pair(ivs[i], numTileLoops));
  }
  return surroundingLoopIVs;
}

SmallVector<int64_t> mlir::affine::extractFromI64ArrayAttr(Attribute attr) {
  if (ArrayAttr array = dyn_cast<ArrayAttr>(attr)) 
    return llvm::to_vector<4>(
        llvm::map_range(array, [](Attribute a) -> int64_t {
          return cast<IntegerAttr>(a).getInt();
        }));
  if (DenseI64ArrayAttr array = dyn_cast<DenseI64ArrayAttr>(attr))
    return SmallVector<int64_t>(array.asArrayRef());

  LLVM_DEBUG(attr.dump());
  llvm_unreachable("Unsupported type for conversion to an integer array.");
}

int mlir::affine::maxConstantInDepthi(
    int depth, SmallVector<Operation *> operationVector,
    DenseMap<Operation *, affine::StatementInfo> &StatementInfoMap) {
  int maxConstant = 0;
  for (auto op : operationVector)
    if (StatementInfoMap.find(op) != StatementInfoMap.end())
      if (((StatementInfoMap[op]).constantRows)[depth] > maxConstant)
        maxConstant = ((StatementInfoMap[op]).constantRows)[depth];
  return maxConstant;
}

size_t mlir::affine::depthOfPerfectlyNestingAroundOp(Operation *op) {
  SmallVector<AffineForOp> loopsAroundOp;
  SmallVector<AffineForOp> perfectlyNestedLoops;
  // get all loops around the op
  getAffineForIVs(*op, &loopsAroundOp);

  if (loopsAroundOp.empty())
    return 0;

  // use outermost loop as the root, get the perfectly nested loop nest from the
  // root
  getPerfectlyNestedLoops(perfectlyNestedLoops, loopsAroundOp[0]);
  
  return perfectlyNestedLoops.size();
}

bool mlir::affine::isControlFlow(Operation *op) {
  return isa<AffineIfOp>(op) || isa<AffineYieldOp>(op) ||
         isa<func::ReturnOp>(op) || isa<AffineForOp>(op) ||
         isa<func::FuncOp>(op) || isa<scf::ForOp>(op) || 
         isa<scf::IfOp>(op) || isa<scf::YieldOp>(op);
}