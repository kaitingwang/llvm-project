//===- PolyTool.cpp --- New Design of the Entire Tool ------ -------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LabelChain.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/PolyToolCorrectionUtils.h"
#include "mlir/Dialect/Affine/PolyToolUtils.h"
#include "mlir/Dialect/Affine/StatementInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Affine/PolyTool.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cloog/cloog.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdio.h>
#include <string>
#include <utility>

using llvm::dbgs;
using namespace std;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::presburger;

#define DEBUG_TYPE "polytool"

void PolyTool::cleanUpMemory() {
  for (auto statementInfo : statementInfoMap)
    for (int i = 0; i < (statementInfo.second).transformationList.size(); i++) {
      delete (statementInfo.second).transformationList[i];
      (statementInfo.second).transformationList[i] = nullptr;
    }
}

bool PolyTool::parallelSecondOuter(Node *node) {
  bool ans = false;
  for (auto child : node->children) {
    ans = (ans || parallelSecondOuter(child));
  }
  if (node && (node->transformation) &&
      (node->transformation)->transformationType == Parallel) {
    if (node->getDepth() == 1)
      ans = true;
  }
  return ans;
}

void PolyTool::findMaxMatrixDimension(func::FuncOp func) {
  // finds the max number of rows and cols accross the whole function
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isControlFlow(op)) {
      if (schedules.find(op) == schedules.end())
        return;
      int tilesize = 0;
      SmallVector<SmallVector<int64_t>> a = IntMatrixToVector(schedules[op][1]);
      SmallVector<SmallVector<int64_t>> b = IntMatrixToVector(schedules[op][0]);
      SmallVector<SmallVector<int64_t>> c = IntMatrixToVector(schedules[op][2]);

      maxSymCount =
          std::max((size_t)maxSymCount, (c.empty() ? 0 : c[0].size()));
      maxRowCount = std::max(a.size() + tilesize, (size_t)(maxRowCount));
      maxRowCount = std::max(b.size() + tilesize, (size_t)(maxRowCount));
      if (!a.empty())
        maxColCount = std::max(a[0].size() + tilesize, (size_t)(maxColCount));
      if (!b.empty())
        maxColCount = std::max(b[0].size() + tilesize, (size_t)(maxColCount));
    }
  });
}

// Gets a FlatAffineRelation inequality from an affine map
static void getInequalityForAffineMap(AffineMap map,
                                      SmallVector<Value> operands, bool isLB,
                                      unsigned idx,
                                      SmallVector<AffineForOp> &loops,
                                      SmallVector<Value> &symbols,
                                      SmallVector<DynamicAPInt> &ineq) {
  AffineExpr expr = map.getResult(0);
  int64_t constant = isLB ? 0 : -1, ivCoeff = isLB ? 1 : -1,
          operandCoeff = isLB ? -1 : 1, specialCoeff = isLB ? -1 : 1;
  // Handle different type of affine expr
  // Make sure not to use local vars
  if (AffineConstantExpr constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    constant += constExpr.getValue();
    for (unsigned i = 0; i < loops.size(); i++) {
      if (i == idx) {
        ineq.push_back(DynamicAPInt(ivCoeff));
      } else {
        ineq.push_back(DynamicAPInt(0));
      }
    }
    for (unsigned i = 0; i < symbols.size(); i++) {
      ineq.push_back(DynamicAPInt(0));
    }
    ineq.push_back(DynamicAPInt(constant * (isLB ? -1 : 1)));
  } else if (AffineBinaryOpExpr binExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    bool hasSpecialOp = false;
    Value SpecialOperand;
    while (binExpr) {

      AffineExpr lhs = binExpr.getLHS(), rhs = binExpr.getRHS();
      AffineExprKind kind = binExpr.getKind();
      AffineConstantExpr constExpr;

      if (!(binExpr = lhs.dyn_cast<AffineBinaryOpExpr>()))
        binExpr = rhs.dyn_cast<AffineBinaryOpExpr>();
      switch (kind) {
      case AffineExprKind::Add: {
        Value operand;
        if (constExpr = lhs.dyn_cast<AffineConstantExpr>()) {
          constant += constExpr.getValue();
        } else if (constExpr = rhs.dyn_cast<AffineConstantExpr>()) {
          constant += constExpr.getValue();
        } else if (AffineSymbolExpr s = lhs.dyn_cast<AffineSymbolExpr>()) {
          operand = operands[map.getNumDims() + s.getPosition()];
          SpecialOperand = operand;
          LLVM_DEBUG(llvm::dbgs() << "FOUND SPECIAL OP" << "\n");
          hasSpecialOp = true;
        } else if (AffineSymbolExpr s = rhs.dyn_cast<AffineSymbolExpr>()) {
          operand = operands[map.getNumDims() + s.getPosition()];
          SpecialOperand = operand;
          LLVM_DEBUG(llvm::dbgs() << "FOUND SPECIAL OP" << "\n");
          hasSpecialOp = true;
        } else
          llvm_unreachable("Unknown affine expr");
        break;
      }
      case AffineExprKind::Mul:
        if (!(constExpr = lhs.dyn_cast<AffineConstantExpr>())) {
          constExpr = rhs.dyn_cast<AffineConstantExpr>();
        }
        operandCoeff *= constExpr.getValue();
        break;
      case AffineExprKind::FloorDiv:
      case AffineExprKind::CeilDiv:
        if (!(constExpr = lhs.dyn_cast<AffineConstantExpr>())) {
          constExpr = rhs.dyn_cast<AffineConstantExpr>();
        }
        ivCoeff *= constExpr.getValue();
        break;
      default:
        break;
      }
      if (!binExpr) {
        Value operand;
        if (AffineDimExpr dim = lhs.dyn_cast<AffineDimExpr>())
          operand = operands[dim.getPosition()];
        else if (AffineDimExpr dim = rhs.dyn_cast<AffineDimExpr>())
          operand = operands[dim.getPosition()];
        else if (AffineSymbolExpr s = lhs.dyn_cast<AffineSymbolExpr>())
          operand = operands[s.getPosition()];
        else if (AffineSymbolExpr s = rhs.dyn_cast<AffineSymbolExpr>())
          operand = operands[s.getPosition()];
        else
          llvm_unreachable("Unknown affine expr");
        for (unsigned i = 0; i < loops.size(); i++) {
          if (i == idx) {
            ineq.push_back(DynamicAPInt(ivCoeff));
          } else if (operand == loops[i].getInductionVar()) {
            ineq.push_back(DynamicAPInt(operandCoeff));
          } else {
            ineq.push_back(DynamicAPInt(0));
          }
        }
        // if we have a index case, go to look at index cast's input
        if (!isa<BlockArgument>(operand))
          if (arith::IndexCastOp iCast =
                  dyn_cast<arith::IndexCastOp>(operand.getDefiningOp()))
            operand = iCast.getIn();
        if (hasSpecialOp && !isa<BlockArgument>(SpecialOperand)) {
          if (arith::IndexCastOp iCast = dyn_cast<arith::IndexCastOp>(
                  SpecialOperand.getDefiningOp())) {
            SpecialOperand = iCast.getIn();
          }
        }
        for (Value s : symbols) {
          if (hasSpecialOp && s == SpecialOperand) {
            ineq.push_back(DynamicAPInt(specialCoeff));
          } else if (s == operand) {
            ineq.push_back(DynamicAPInt(operandCoeff));
          } else {
            ineq.push_back(DynamicAPInt(0));
          }
        }
        ineq.push_back(DynamicAPInt(constant * (isLB ? -1 : 1)));
      }
    }
  } else {
    Value operand;
    if (AffineDimExpr dimExpr = expr.dyn_cast<AffineDimExpr>()) {
      operand = operands[dimExpr.getPosition()];
    } else {
      AffineSymbolExpr symExpr = expr.dyn_cast<AffineSymbolExpr>();
      operand = operands[symExpr.getPosition()];
    }

    constant = 1;
    for (unsigned i = 0; i < loops.size(); i++) {
      if (i == idx) {
        ineq.push_back(DynamicAPInt(ivCoeff));
      } else if (operand == loops[i].getInductionVar()) {
        ineq.push_back(DynamicAPInt(operandCoeff));
      } else {
        ineq.push_back(DynamicAPInt(0));
      }
    }
    // if we have a index case, go to look at index cast's input
    if (!isa<BlockArgument>(operand))
      if (arith::IndexCastOp iCast =
              dyn_cast<arith::IndexCastOp>(operand.getDefiningOp()))
        operand = iCast.getIn();
    for (Value s : symbols) {
      if (s == operand) {
        ineq.push_back(DynamicAPInt(operandCoeff));
      } else {
        ineq.push_back(DynamicAPInt(0));
      }
    }
    ineq.push_back(DynamicAPInt(isLB ? 0 : -1));
  }
}

static int parseIntegerSet(int numDims, AffineExpr expr,
                           SmallVector<Value> operands,
                           SmallVector<AffineForOp> loops,
                           SmallVector<Value> symbols,
                           SmallVector<DynamicAPInt> &ineq, int literal) {
  LLVM_DEBUG(expr.dump());
  if (AffineConstantExpr constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    if (literal == 0)
      literal = 1;
    return literal * constExpr.getValue();
  } else if (AffineBinaryOpExpr binExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    AffineExprKind kind = binExpr.getKind();
    AffineExpr lhs = binExpr.getLHS(), rhs = binExpr.getRHS();
    AffineConstantExpr constExpr;
    LLVM_DEBUG(llvm::dbgs() << "NUMBER OF DIMENSIONS: " << numDims << "\n");
    LLVM_DEBUG(lhs.dump());
    LLVM_DEBUG(rhs.dump());
    switch (kind) {
    case AffineExprKind::Add: {
      // LLVM_DEBUG(llvm::dbgs() << "INSIDE ADD: " << numDims << "\n");
      int cst1 = parseIntegerSet(numDims, lhs, operands, loops, symbols, ineq,
                                 literal);
      // LLVM_DEBUG(llvm::dbgs() << "INSIDE ADD #1: " << numDims << "\n");
      int cst2 = parseIntegerSet(numDims, rhs, operands, loops, symbols, ineq,
                                 literal);
      // LLVM_DEBUG(llvm::dbgs() << "INSIDE ADD #2: " << numDims << "\n");
      return cst1 + cst2;
      break;
    }
    case AffineExprKind::Mul: {
      int literalNew = parseIntegerSet(numDims, rhs, operands, loops, symbols,
                                       ineq, literal);
      int cst = parseIntegerSet(numDims, lhs, operands, loops, symbols, ineq,
                                literalNew);
      return cst;
      break;
    }
    default:
      break;
    }
  }
  Value operand;
  if (AffineDimExpr dim = expr.dyn_cast<AffineDimExpr>()) {
    LLVM_DEBUG(llvm::dbgs() << "INSIDE DIM #1: " << ineq.size() << "\n");
    operand = operands[dim.getPosition()];
    LLVM_DEBUG(llvm::dbgs() << "INSIDE DIM #2: " << ineq.size() << "\n");
    for (int i = 0; i < loops.size(); i++) {
      LLVM_DEBUG(llvm::dbgs() << "INSIDE DIM #3: " << loops.size() << " " << i
                              << " " << ineq.size() << "\n");
      if (operand == loops[i].getInductionVar()) {
        LLVM_DEBUG(llvm::dbgs() << ":FOUND\n");
        ineq[i] = DynamicAPInt(literal);
        if (literal == 0) {
          ineq[i] = DynamicAPInt(1);
        }
        // return literal;
      }
    }
    return 0;
  } else if (AffineSymbolExpr s = expr.dyn_cast<AffineSymbolExpr>()) {
    LLVM_DEBUG(llvm::dbgs() << "INSIDE SYMBOL #1: " << s.getPosition() << "\n");
    operand = operands[numDims + s.getPosition()];
    LLVM_DEBUG(llvm::dbgs() << operand << "\n");
    int cnt = numDims + s.getPosition();
    ineq[cnt] = DynamicAPInt(literal);
    if (literal == 0) {
      ineq[cnt] = DynamicAPInt(1);
    }
  } else
    llvm_unreachable("Unknown affine expr");
  return 0;
}

void PolyTool::findStatementInfo(ModuleOp module, OpBuilder b) {
  int numDim = 0, numLocal = 0;
  DenseMap<mlir::Value, bool> memrefVals;
  DenseMap<mlir::Value, bool> numSym;
  DenseMap<mlir::Value, bool> numSymInsideLoop;
  // SmallVector<Operation *, 8> statements;
  DenseMap<mlir::Value, SmallVector<Operation *>> readMap, writeMap;
  Operation *prevOp = nullptr;
  SmallVector<SmallVector<DynamicAPInt, 8>, 8> integerSamples;

  module.walk<mlir::WalkOrder::PreOrder>([&](func::FuncOp func) {
    if (func.getOperation()->hasAttr("skip.poly")) {
      LLVM_DEBUG(dbgs() << "Skipping function with skip.poly attribute: "
                        << func.getName() << "\n");
      return;
    }
    if (func != NULL) {
      assert(this->targetFunc == nullptr &&
             "Multiple kernel functions found, should only have one!");

      this->targetFunc = func;
      // Parameters
      SmallVector<Value> symbols; // records all symbols in the function
      for (Value v : func.getRegion().getArguments())
        if (v.getType().isIntOrIndex()) {
          symbols.push_back(v);
        }
      this->symbols = symbols;
      func->setAttr("PolyTool", b.getUnitAttr());
    }
    func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      numDim = mlir::affine::getNestingDepth(op) > numDim
                   ? mlir::affine::getNestingDepth(op)
                   : numDim;
      if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
        addOptoMap(writeMap, storeOp.getMemref(), storeOp);
        addOptoMap(readMap, storeOp.getValue(), storeOp);
        memrefVals[storeOp.getMemref()] = true;
        recordScheduleInfo(statements, constants, scheduleOrder, prevOp, op,
                           stmtCount);
      } else if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
        addOptoMap(writeMap, loadOp.getResult(), loadOp);
        addOptoMap(readMap, loadOp.getMemref(), loadOp);
        memrefVals[loadOp.getMemref()] = true;
        recordScheduleInfo(statements, constants, scheduleOrder, prevOp, op,
                           stmtCount);
      } else if (!isa<AffineYieldOp, AffineForOp, AffineIfOp, func::ReturnOp, func::FuncOp>(op))
        recordScheduleInfo(statements, constants, scheduleOrder, prevOp, op,
                           stmtCount);

      OpBuilder b(func.getContext());
      SmallVector<AffineForOp> loops;
      SmallVector<Value> symbols; // records all symbols in the function
      unsigned depth = getDepthAndLoopOperands(op, loops, symbols, b);
      for (auto sym : symbols)
        numSymInsideLoop[sym] = true;
    });
  });
  computeDependencePolyhedron(dependenceEdges, readMap, writeMap, memrefVals);

  schedules = initSchedule(statements, numSymInsideLoop.size());
  module.walk<mlir::WalkOrder::PreOrder>(
      [&](func::FuncOp func) { findMaxMatrixDimension(func); });
  module.walk<mlir::WalkOrder::PreOrder>([&](func::FuncOp func) {
    if (func.getOperation()->hasAttr("skip.poly")) {
      LLVM_DEBUG(dbgs() << "Skipping function with skip.poly attribute: "
                        << func.getName() << "\n");
      return;
    }
    func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      if (!isControlFlow(op))
        if (schedules.find(op) != schedules.end())
          addStatementInfo(module, func, op, constants[op],
                           IntMatrixToVector(schedules[op][1]),
                           IntMatrixToVector(schedules[op][0]),
                           IntMatrixToVector(schedules[op][2]));
    });
  });
}

void PolyTool::addStatementInfo(ModuleOp module, func::FuncOp func,
                                Operation *s, SmallVector<int> constantRow,
                                SmallVector<SmallVector<int64_t>> a,
                                SmallVector<SmallVector<int64_t>> b,
                                SmallVector<SmallVector<int64_t>> c) {
  statementInfoMap[s] = StatementInfo();
  for (auto num : constantRow)
    (statementInfoMap[s].constantRows).push_back(num);

  statementInfoMap[s].opId = opCount;
  statementInfoMap[s].maxSymCount = maxSymCount;
  statementInfoMap[s].maxTileSize = maxTileSize;
  statementInfoMap[s].maxColCount = maxColCount;
  statementInfoMap[s].maxRowCount = maxRowCount;
  statementInfoMap[s].a = a;
  statementInfoMap[s].b = b;
  statementInfoMap[s].c = c;
  opCount++;
  statementInfoMap[s].domain = findDomain(module, func, s, c, a);
  statementInfoMap[s].scattering = findScattering(module, func, s, a, b, c);
  int depth = getNumCommonSurroundingLoops(*s, *s);
  for (int i = 0; i < depth; i++) {
    (statementInfoMap[s].isIteratorTiled).push_back(false);
  }
  affine::StatementInfo st = statementInfoMap[s];

  LLVM_DEBUG(
    llvm::dbgs() << "Printing recorded schedule of the op:\n";
    s->dump();
    st.printConstantRow();
    st.printDomain();
    st.printScattering();
  );
}

SmallVector<SmallVector<int64_t>>
PolyTool::findDomain(ModuleOp module, func::FuncOp func, Operation *op,
                     SmallVector<SmallVector<int64_t>> c,
                     SmallVector<SmallVector<int64_t>> a) {

  OpBuilder b1(module.getContext());
  OpBuilder b(func.getContext());

  unsigned numLoops = 0, numOps = 0;
  FlatLinearConstraints flat;

  numLoops++;

  LLVM_DEBUG(llvm::dbgs() << "Domain Printer Starts\n");
  // Domain
  SmallVector<AffineForOp> loops;
  SmallVector<AffineIfOp> ifs;
  // get parent loop nest depth, ivs and operands
  unsigned depth = getDepthAndLoopOperands(op, loops, symbols, b);
  FlatLinearConstraints cst(depth * 2, 0, loops.size() + symbols.size() + 1,
                            loops.size(), symbols.size(), 0);
  // build inequalities from the affine maps in the loop ub and lb for each
  // loop
  for (unsigned i = 0; i < loops.size(); i++) {
    SmallVector<DynamicAPInt> ineqLB, ineqUB;
    getInequalityForAffineMap(loops[i].getLowerBoundMap(),
                              loops[i].getLowerBoundOperands(), true, i, loops,
                              symbols, ineqLB);
    getInequalityForAffineMap(loops[i].getUpperBoundMap(),
                              loops[i].getUpperBoundOperands(), false, i, loops,
                              symbols, ineqUB);
    cst.addInequality(ineqLB);
    cst.addInequality(ineqUB);
  };
  unsigned IfDepth = getDepthAndIfOperands(op, ifs, symbols, b);
  for (unsigned i = 0; i < ifs.size(); i++) {
    SmallVector<DynamicAPInt> ineq;
    IntegerSet affineCondition = (ifs[i].getIntegerSet());
    auto expr = affineCondition.getConstraint(0);
    SmallVector<Value> operands(ifs[i].getOperands());
    auto numDims = affineCondition.getNumDims();
    for (int i = 0; i < loops.size(); i++)
      ineq.push_back(DynamicAPInt(0));

    for (Value s : symbols)
      ineq.push_back(DynamicAPInt(0));

    int finalCst =
        parseIntegerSet(numDims, expr, operands, loops, symbols, ineq, 0);
    ineq.push_back(DynamicAPInt(finalCst));
    cst.addInequality(ineq);
    LLVM_DEBUG(expr.dump());
  }
  flat = cst;
  LLVM_DEBUG(llvm::dbgs() << "Domain Printer ends\n");
  return FlatConstraintToVector(flat);
}

SmallVector<SmallVector<int64_t>>
PolyTool::findScattering(ModuleOp module, func::FuncOp func, Operation *op,
                         SmallVector<SmallVector<int64_t>> a,
                         SmallVector<SmallVector<int64_t>> b,
                         SmallVector<SmallVector<int64_t>> c) {
  LLVM_DEBUG(llvm::dbgs() << "Scatter Printer starts\n");
  // Scattering

  // in case we don't have any iterators in the program
  // there is no loops inside the code so there should
  // be no scatterings at all
  if (maxRowCount == 0) {
    SmallVector<SmallVector<int64_t>> tmp;
    return tmp;
  }
  int numOrderingVar = maxRowCount + 1;
  int curDepth = 0;
  bool activeDistribute = false;
  int actualDepth = curDepth;
  SmallVector<SmallVector<int64_t>> combined = matmul(a, b);

  int numIter = (combined.empty() ? 0 : combined[0].size());
  int numIterRows = combined.size();
  LLVM_DEBUG(llvm::dbgs() << "MAX COL: " << maxColCount << "\n");
  LLVM_DEBUG(llvm::dbgs() << "MAX ROW: " << maxRowCount << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NUM Iter: " << numIter << "\n");
  LLVM_DEBUG(llvm::dbgs() << "maxSymCount: " << maxSymCount << "\n");
  LLVM_DEBUG(llvm::dbgs() << "NUM Ordering Var: " << numOrderingVar << "\n");
  int numCol = maxRowCount + numIter + maxSymCount;

  numCol += numOrderingVar;

  FlatLinearConstraints cst(0, maxRowCount + numOrderingVar, numCol,
                            maxRowCount + numIter + numOrderingVar,
                            maxSymCount - 1, 0);
  {
    SmallVector<DynamicAPInt> eq;
    // the row for c0, or S1
    for (int j = 0; j < numOrderingVar; j++)
      eq.push_back(DynamicAPInt(j == 0 ? 1 : 0)); // multi dim ordering
    for (int j = 0; j < maxColCount; j++)
      eq.push_back(DynamicAPInt(0));

    for (int j = 0; j < numIter; j++)
      eq.push_back(DynamicAPInt(0));
    for (int j = 0; j < maxSymCount - 1; j++)
      eq.push_back(DynamicAPInt(0));
    // Fill the constants with 0 as default value
    while (eq.size() < numCol)
      eq.push_back(DynamicAPInt(0));
    cst.addEquality(eq);
  }
  for (unsigned i = 0; i < maxRowCount; i++) {
    SmallVector<DynamicAPInt> eq;
    // A diagonal matrix for c1 ... cn with ordering
    for (int j = 0; j < maxColCount + numOrderingVar; j++) {
      if (i * 2 + 1 == j)
        eq.push_back(DynamicAPInt(1));
      else
        eq.push_back(DynamicAPInt(0));
    }
    // copy of matrix a*b
    for (int j = 0; j < numIter; j++) {
      if (i >= combined.size())
        eq.push_back(DynamicAPInt(0));
      else
        eq.push_back(DynamicAPInt(-1 * combined[i][j]));
    }

    // copy of matrix c
    // In this stage, since no transformation has been applied
    // all the c matrix must be equal to zero so it is safe
    // to replace them with zero anyway so I did the sanity
    // check to avoid crashes
    for (int j = 0; j < maxSymCount; j++) {
      if (i >= c.size() || j >= c[i].size())
        eq.push_back(DynamicAPInt(-1 * 0));
      else
        eq.push_back(DynamicAPInt(-1 * c[i][j]));
    }
    cst.addEquality(eq);
    SmallVector<DynamicAPInt> eqOrdering;
    for (int j = 0; j < maxColCount + numOrderingVar; j++)
      eqOrdering.push_back(
          DynamicAPInt(j == 1 + i * 2 + 1 ? 1 : 0)); // multi dim ordering

    for (int j = 0; j < numIter; j++)
      eqOrdering.push_back(DynamicAPInt(0));

    for (int j = 0; j < maxSymCount - 1; j++)
      eqOrdering.push_back(DynamicAPInt(0));

    // Fill the constants with 0 as default value
    while (eqOrdering.size() < numCol) {
      eqOrdering.push_back(DynamicAPInt(0));
    }
    cst.addEquality(eqOrdering);
  }
  LLVM_DEBUG(llvm::dbgs() << "Scatter Printer ends\n");

  return FlatConstraintToVector(cst);
}

void PolyTool::applyTransformations(ModuleOp module, OpBuilder b) {
  const RaggedArray<mlir::transform::MappedValue> extraMappings;
  for (auto op : module.getBody()->getOps<transform::TransformOpInterface>()) {
    transform::validatorApplyTransforms(
        module, op, extraMappings,
        transform::TransformOptions().enableExpensiveChecks(false), &schedules,
        &dependenceEdges, &constants, &distributeParent, &statementInfoMap,
        &tgMap, tgRoot, &minTile, &maxTile);
  }
  LLVM_DEBUG(llvm::dbgs() << "Dependency edges after transform ops:\n");
  LLVM_DEBUG(printDependenceEdges(dependenceEdges));
}

LogicalResult PolyTool::applyCorrection(bool changedIsIteratorTiled) {
  // In the case that we are not tiling anything
  // but we are trying to parallelize the second
  // outermost loops we need to fake tiling for
  // the correction and bring it back immediately
  // after correction for the codeGen.
  // check seidal-2d for an example
  if (changedIsIteratorTiled) {
    applyFakeTilingCase(true);
    for (int i = 0; i < dependenceEdges.size(); i++) {
      dependenceEdges[i].parallelDepth = -1;
    }
  }
  int numLocal = 0, currIter = 0, edgei = 0;
  int numDim = maxRowCount - maxTileSize;
  bool failedCorrection = false;
  SmallVector<SmallVector<DynamicAPInt, 8>, 8> integerSamples;

  extractCandidateSchedule(schedules, constants, statementInfoMap,
                           symbols.size(), numDim, minTile, maxTile);
  printSchedules(schedules, constants, scheduleOrder, minTile, maxTile);

  while (edgei < dependenceEdges.size()) {
    dependenceEdge e = dependenceEdges[edgei];
    if (statementInfoMap[e.src].constantRows[0] <
        statementInfoMap[e.dst].constantRows[0]) {
      dependenceEdges[edgei].isEmpty = true;
    } else if (statementInfoMap[e.src].constantRows[0] >
               statementInfoMap[e.dst].constantRows[0]) {
      // First constant row violation is non-correctable
      failedCorrection = true;
      break;
    }
    edgei += 1;
  }
  while (!allPolyEmpty(dependenceEdges) && currIter < numDim &&
         !failedCorrection) {
    LLVM_DEBUG(dbgs() << "Correction start (dimension=" << currIter
                      << "): " << numDim << " " << symbols.size() << " "
                      << numLocal << "\n");
    SmallVector<DynamicAPInt, 8> integerSample =
        correctIter(statements, schedules, constants, dependenceEdges,
                    scheduleOrder, currIter, stmtCount + 1, numDim,
                    symbols.size(), numLocal, minTile, maxTile, integerSamples);
    if (integerSample.empty()) {
      failedCorrection = true;
      break;
    }
    for (Operation *s : statements) {
      LLVM_DEBUG(dbgs() << "Update schedule for stmt:"; s->dump();
                 dbgs() << "\n");
      int curNumDim = schedules[s][1].getNumRows();
      if (curNumDim > currIter) { // TODO: Should we break here?
        for (unsigned j = 0; j < schedules[s][1].getNumColumns(); j++)
          schedules[s][1](currIter, j) = int64_t(
              integerSample[scheduleOrder[s] * (numDim + symbols.size() + 1) +
                            j]);
        for (unsigned j = 0; j <= symbols.size(); j++)
          schedules[s][2](currIter, j) = int64_t(
              integerSample[scheduleOrder[s] * (numDim + symbols.size() + 1) +
                            j + numDim]);
      }
    }
    currIter += 1;
  }
  if (failedCorrection) {
    dbgs()
        << "<INFO> Schedule has dependency violation. Correction() failed.\n";
    return failure();
  }
  // Always print the final schedule
  bool noCorrection = true;
  for (Operation *s : statements) {
    bool isStmtCorrected = false;
    if (schedules[s][0].getNumRows() == 0 ||
        schedules[s][1].getNumRows() == 0 || schedules[s][2].getNumRows() == 0)
      continue;
    SmallVector<SmallVector<int64_t>> a = IntMatrixToVector(schedules[s][1]);
    SmallVector<SmallVector<int64_t>> b = IntMatrixToVector(schedules[s][0]);
    SmallVector<SmallVector<int64_t>> c = IntMatrixToVector(schedules[s][2]);
    SmallVector<SmallVector<int64_t>> combined = matmul(a, b);
    LLVM_DEBUG(llvm::dbgs() << "matmul(a,b): \n");
    for (auto i : combined) {
      for (auto j : i) {
        LLVM_DEBUG(llvm::dbgs() << j << " ");
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
    }

    size_t perfectLoopNestDim = depthOfPerfectlyNestingAroundOp(s);

    if (perfectLoopNestDim >= 2 && changedIsIteratorTiled) {
      applyFakeTilingCase(false);
    }

    if (statementInfoMap[s].manualSetFlag) {
      statementInfoMap[s].a = a;
      statementInfoMap[s].applyUserSchedule();
      a = statementInfoMap[s].a;
    }

    if (!(schedules[s][1].isIdentityWithZeroRows() &&
          schedules[s][2].isEmpty())) {
      isStmtCorrected = true; // correction happened

      // Then we need to inject correction into domain and scattering
      injectCandidateSchedule(combined, c, constants[s], statementInfoMap[s],
                              symbols.size(), minTile[s], maxTile[s]);
      injectCandidateDomain(a, statementInfoMap[s]);
    }
    if (perfectLoopNestDim >= 2 && changedIsIteratorTiled) {
      statementInfoMap[s].applySumTrick(0);
    }

    // handle post tiling primitives when corection took place
    // post tiling reorder only needs special treatment when correction happened
    if (isStmtCorrected)
      statementInfoMap[s].replayPostTilingPrimitives();

    noCorrection &= !isStmtCorrected;
  }
  if (noCorrection)
    dbgs()
        << "<INFO> Schedule has no dependency violation. Check() completes. \n";
  else {
    dbgs() << "<INFO> Schedule has dependency violation. Correction() "
              "completes. \n";
  }
  return success();
}

bool PolyTool::isFakeTilingCase() {
  ModuleOp module = getOperation();
  OpBuilder b(module.getContext());
  bool ans = true;
  module.walk<mlir::WalkOrder::PreOrder>([&](func::FuncOp func) {
    if (func.getOperation()->hasAttr("skip.poly")) {
      LLVM_DEBUG(dbgs() << "Skipping function with skip.poly attribute: "
                        << func.getName() << "\n");
      return;
    }

    func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      if (statementInfoMap[op].isTiled())
        ans = false;
    });
  });
  ans = ans && parallelSecondOuter(tgRoot);
  return ans;
}

void PolyTool::applyFakeTilingCase(bool apply) {
  ModuleOp module = getOperation();
  OpBuilder b(module.getContext());
  module.walk<mlir::WalkOrder::PreOrder>([&](func::FuncOp func) {
    if (func.getOperation()->hasAttr("skip.poly")) {
      LLVM_DEBUG(dbgs() << "Skipping function with skip.poly attribute: "
                        << func.getName() << "\n");
      return;
    }
    func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      if (statementInfoMap.find(op) != statementInfoMap.end()) {
        size_t perfectLoopNestDim = depthOfPerfectlyNestingAroundOp(op);
        // if the op does not have a perfect loop nest of dimension at least 2,
        // it can't be tiled like this
        if (perfectLoopNestDim >= 2)
          for (unsigned i = 0;
               i < std::min(perfectLoopNestDim,
                            statementInfoMap[op].isIteratorTiled.size());
               i++) {
            statementInfoMap[op].isIteratorTiled[i] = apply;
          }
      }
    });
  });
}

void PolyTool::runOnOperation() {

  ModuleOp module = getOperation();
  OpBuilder b(module.getContext());

  // Phase1: Find the domain, scattering and constant rows for all the
  // statements in the module
  LLVM_DEBUG(llvm::dbgs() << "PHASE 1 BEGINS\n");
  findStatementInfo(module, b);
  LLVM_DEBUG(llvm::dbgs() << "PHASE 1 ENDS\n");

  // Phase2: Apply transformations to the statement's domain and scattering
  // matrices
  LLVM_DEBUG(llvm::dbgs() << "PHASE 2 BEGINS\n");
  applyTransformations(module, b);
  LLVM_DEBUG(llvm::dbgs() << "PHASE 2 ENDS\n");

  // In the case that we are not tiling anything
  // but we are trying to parallelize the second
  // outermost loops we need to fake tiling for
  // the correction and bring it back immediately
  // after correction for the codeGen.
  // check seidal-2d for an example
  bool changedIsIteratorTiled = isFakeTilingCase();

  // Apply correction
  if (failed(applyCorrection(changedIsIteratorTiled)))
    return signalPassFailure();

  // Phase 3
  LLVM_DEBUG(llvm::dbgs() << "PHASE 3 BEGINS\n");
  generateOptimizedCode();
  LLVM_DEBUG(llvm::dbgs() << "PHASE 3 ENDS\n");

  LLVM_DEBUG(llvm::dbgs() << "CLEAN UP MEMORY BEGINS\n");
  cleanUpMemory();
  LLVM_DEBUG(llvm::dbgs() << "CLEAN UP MEMORY ENDS\n");
}

std::unique_ptr<Pass> mlir::affine::createPolyToolPass() {
  return std::make_unique<PolyTool>();
}

std::unique_ptr<Pass> mlir::affine::createPolyToolPass(int parallelDimensions,
                                                       bool noThreadLocals,
                                                       bool polybenchCodeGen,
                                                       std::string codeGenOutputFile) {
  return std::make_unique<PolyTool>(parallelDimensions, noThreadLocals, polybenchCodeGen, codeGenOutputFile);
}
