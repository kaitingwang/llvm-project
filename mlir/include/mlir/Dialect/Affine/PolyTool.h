//===- PolyTool.h - The main class of the PolyTool --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This header defines all the classes for PolyTool Controller.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_POLYTOOL_H
#define MLIR_DIALECT_AFFINE_POLYTOOL_H

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/PolyToolCorrectionUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <cloog/cloog.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <string>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_POLYTOOL
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

using namespace std;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::presburger;

struct PolyTool : public affine::impl::PolyToolBase<PolyTool> {
  PolyTool() = default;
  explicit PolyTool(uint64_t parallelDimensions, bool noThreadLocals, bool polybenchCodeGen, std::string codeGenOutputFile) {
    this->parallelDimensions = parallelDimensions;
    this->noThreadLocals = noThreadLocals;
    this->polybenchCodeGen = polybenchCodeGen;
    this->codeGenOutputFile = codeGenOutputFile;
  }
  ~PolyTool() { tgRoot->freeTree(); }

  // The controller inside the MVC design patterm
  // this function is responsible for running all
  // three phases and calling functions in the
  // correct order
  void runOnOperation() override;

private:
  // to be able to run the transform validator we need the following
  SmallVector<dependenceEdge, 8> dependenceEdges;
  SmallVector<Operation *, 8> statements;
  DenseMap<Operation *, int> scheduleOrder, minTile, maxTile;
  DenseMap<mlir::affine::AffineForOp, mlir::affine::AffineForOp>
      distributeParent;
  DenseMap<Operation *, SmallVector<int>> constants;
  DenseMap<Operation *, SmallVector<IntMatrix, 4>> schedules;
  DenseMap<Operation *, affine::StatementInfo> statementInfoMap;
  DenseMap<Value, mlir::affine::Node *> tgMap;
  Node *tgRoot = new Node();
  int maxSymCount = 0;
  int maxTileSize = 0;
  int maxColCount = 0;
  int opCount = 0;
  int maxRowCount = 0;
  int stmtCount = -1;
  SmallVector<Value>
      symbols; // all symbols (func arguments) in the target function
  func::FuncOp targetFunc; // the target function to be optimized

  // find maxSymCount, maxTileSize, maxColCount, maxRowcount before finding the
  // scattering This information is later used to make the number of rows in
  // scattering the same
  void findMaxMatrixDimension(func::FuncOp func);

  // Finds the information (Domain/Scattering/Constant Rows) for a single
  // operation op inside function func and module module
  void addStatementInfo(ModuleOp module, func::FuncOp func, Operation *s,
                        SmallVector<int> constantRow,
                        SmallVector<SmallVector<int64_t>> a,
                        SmallVector<SmallVector<int64_t>> b,
                        SmallVector<SmallVector<int64_t>> c);

  // Finds the information (Domain/Scattering/Constant Rows) of all statements
  // inside the module
  void findStatementInfo(ModuleOp module, OpBuilder b);

  // Returns the domain matrix for a single operation Op inside the function
  // func in module module
  SmallVector<SmallVector<int64_t>>
  findDomain(ModuleOp module, func::FuncOp func, Operation *op,
             SmallVector<SmallVector<int64_t>> c,
             SmallVector<SmallVector<int64_t>> a);

  // Returns the scattering matrix for a single operation Op inside the function
  // func in module module
  SmallVector<SmallVector<int64_t>>
  findScattering(ModuleOp module, func::FuncOp func, Operation *op,
                 SmallVector<SmallVector<int64_t>> a,
                 SmallVector<SmallVector<int64_t>> b,
                 SmallVector<SmallVector<int64_t>> c);
  void applyTransformations(ModuleOp module, OpBuilder b);
  LogicalResult applyCorrection(bool);
  bool isFakeTilingCase();
  bool parallelSecondOuter(Node *node);
  void applyFakeTilingCase(bool);

  // C code generation part using CLooG
  void generateOptimizedCode();

  // Parse MLIR operations (load/compute/store) to C++ statements
  void cleanUpMemory();
};

#endif // MLIR_DIALECT_AFFINE_LABELCHAIN_H
