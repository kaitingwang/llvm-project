//===- PolyToolPrinterUtil.cpp - Utilities for PolyTool code generation -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/PolyToolPrinterUtils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LabelChain.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/PolyTool.h"
#include "mlir/Dialect/Affine/PolyToolUtils.h"
#include "mlir/Dialect/Affine/StatementInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cloog/cloog.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

// This piece of code is modeled as a few passes.
// Every pass takes a code fragment, a context and
// a target function and returns an extended code fragment.

using namespace std;
using namespace mlir;
using namespace mlir::affine;

#define DEBUG_TYPE "polytool"

/*
Utility Functions
*/

std::string llvmTypeToCType(const mlir::Type &type) {
  // Note, PyDSL currently lowers all ints to signless
  // integers.
  if (type.isIndex()) {
    return "size_t ";
  } else if (type.isF32()) {
    return "float ";
  } else if (type.isF64()) {
    return "double ";
  } else if (type.isSignlessInteger(8)) {
    return "int8_t ";
  } else if (type.isSignlessInteger(16)) {
    return "int16_t ";
  } else if (type.isSignlessInteger(32)) {
    return "int ";
  } else if (type.isSignlessInteger(64)) {
    return "int64_t ";
  } else {
    // I appologise in advance for not providing more debug info
    type.dump();
    llvm::report_fatal_error("Polytool: Could not convert Type to String");
  }
}

std::string UnparserState::declget(const mlir::Value &v) {
  if (valueToName.find(v) != valueToName.end()) {
    return valueToName[v];
  }

  mlir::Type tpe = v.getType();
  std::string newVarName = "var" + std::to_string(varCount++);

  // We assume no memref variables.
  varDefs += llvmTypeToCType(tpe) + newVarName + ";\n";

  valueToName[v] = newVarName;
  return newVarName;
}

void UnparserState::defineStatement(std::string statementBody,
                                    int numContainingLoops) {
  std::string statement = "#define S" + std::to_string(++statementCount) + "(";
  for (int i = 0; i < numContainingLoops; ++i) {
    if (i)
      statement += ", ";
    statement += "i" + std::to_string(i);
  }
  statement += ") " + statementBody + ";\n";

  statementDefs += statement;
}

// unparses affine map into C++ indicies

std::string
parseAffineExpression(AffineExpr expr, const SmallVector<Value> &operands,
                      SmallVector<std::pair<Value, int>> surroundingLoopIVs,
                      int numDims, UnparserState &ctx) {
  if (AffineBinaryOpExpr bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    return "(" +
           parseAffineExpression(bin.getLHS(), operands, surroundingLoopIVs,
                                 numDims, ctx) +
           (expr.getKind() == AffineExprKind::Add ? " + " : " * ") +
           parseAffineExpression(bin.getRHS(), operands, surroundingLoopIVs,
                                 numDims, ctx) +
           ")";
  } else if (AffineConstantExpr constExpr =
                 dyn_cast<AffineConstantExpr>(expr)) {
    return std::to_string(constExpr.getValue());
  } else if (AffineSymbolExpr symbol = dyn_cast<AffineSymbolExpr>(expr)) {
    return ctx.valueToName[operands[numDims + symbol.getPosition()]];
  } else if (AffineDimExpr dim = dyn_cast<AffineDimExpr>(expr)) {
    size_t ivIdx = 0;

    assert(surroundingLoopIVs.size() > 0 &&
           "Dimension expr must point to a loop, but no loop found!");
    for (const std::pair<Value, int> ivInfo : surroundingLoopIVs) {
      if (ivInfo.first == operands[dim.getPosition()]) {
        ivIdx += ivInfo.second;
        break;
      }
      ivIdx++;
    }
    // we have at most <number of actual loops> + <number of tile loop> iterator
    assert(surroundingLoopIVs[surroundingLoopIVs.size() - 1].second +
                   surroundingLoopIVs.size() >
               ivIdx &&
           "Dimension expr did not match any loop around the op!");
    return "i" + std::to_string(ivIdx);
  } else {
    llvm_unreachable("unknown affine expression value");
  }
}

/// builds index for read/write from/to an array in the format of:
/// [stride_i * index_i + ...] for dynamic memref or
/// [index_1][index_2][...
static std::string
assembleMemRefIndex(bool isDynamicMemref, size_t numResultsInMap,
                    const std::string &memrefName,
                    const SmallVector<std::string> &parsedIndex) {
  std::string assembledIndex = "[";
  for (size_t i = 0; i < numResultsInMap; ++i) {
    if (isDynamicMemref) {
      if (i)
        assembledIndex += " + ";
      std::string strideArg = memrefName;
      strideArg += "_stride_" + std::to_string(i);
      assembledIndex += "(" + parsedIndex[i] + " * " + strideArg + ")";
    } else {
      assembledIndex += parsedIndex[i];
      if (i < numResultsInMap - 1)
        assembledIndex += "][";
    }
  }
  return assembledIndex + "]";
}

static std::string parseAffineMap(
    Operation *op,
    llvm::SmallVector<std::pair<mlir::Value, int>> surroundingLoopIVs,
    UnparserState &ctx) {
  mlir::AffineMap map;
  llvm::SmallVector<mlir::Value> operands;
  mlir::Value value;
  bool isDynamicMemref;
  if (AffineStoreOp store = dyn_cast<AffineStoreOp>(op)) {
    map = store.getAffineMapAttr().getAffineMap();
    operands = store.getIndices();
    value = store.getMemRef();
  } else if (AffineLoadOp load = dyn_cast<AffineLoadOp>(op)) {
    map = load.getAffineMapAttr().getAffineMap();
    operands = load.getIndices();
    value = load.getMemRef();
  }
  if (mlir::MemRefType mrtpe = dyn_cast<mlir::MemRefType>(value.getType())) {
    isDynamicMemref = mlir::ShapedType::isDynamicShape(mrtpe.getShape()) &&
                      !ctx.polybenchCodegen;
  }
  SmallVector<std::string> parsedIndex(map.getNumResults());
  for (size_t i = 0; i < map.getNumResults(); ++i)
    parsedIndex[i] = parseAffineExpression(
        map.getResult(i), operands, surroundingLoopIVs, map.getNumDims(), ctx);

  return assembleMemRefIndex(isDynamicMemref, map.getNumResults(),
                             ctx.valueToName[value], parsedIndex);
}

static bool shouldPrintOperation(Operation *op) {
  return isa<memref::AllocaOp, AffineStoreOp, memref::StoreOp>(op);
}

// Prints the .cloog file containing all domain, scattering and constants
// Copied from the old implementation
void printOpenScopToFile(UnparserState &ctx) {
  std::string fileName = ctx.cloogFileName;
  std::error_code ec;
  llvm::raw_fd_ostream scopStream(llvm::StringRef(fileName), ec);

  scopStream << "c"
             << "\n\n"; // c for C code

  // print the number and the matrix of the arguments of the function
  int numIntegerArgs = 0;
  for (auto arg : ctx.targetFunc.getRegion().getArguments()) {
    if (arg.getType().isIntOrIndex())
      ++numIntegerArgs;
  }

  FlatLinearConstraints argsCst(numIntegerArgs, 0, numIntegerArgs + 1,
                                numIntegerArgs, 0, 0);
  for (int i = 0; i < numIntegerArgs; i++) {
    SmallVector<DynamicAPInt> ineq;
    for (int j = 0; j < numIntegerArgs; j++) {
      if (j == i)
        ineq.push_back(DynamicAPInt(1));
      else
        ineq.push_back(DynamicAPInt(0));
    }
    ineq.push_back(DynamicAPInt(0));
    argsCst.addInequality(ineq);
  }
  printConstraints(argsCst, scopStream);

  scopStream << "1\n";
  // print arguments of this function
  int varNameCount = 0;
  for (Value v : ctx.targetFunc.getRegion().getArguments()) {
    varNameCount++;
    if (!v.getType().isIntOrIndex())
      continue;
    std::string name("arg");
    name += std::to_string(varNameCount);
    scopStream << name << " ";
  }
  scopStream << " \n\n";

  unsigned numOps = ctx.statementCount;

  // print the domain
  LLVM_DEBUG(llvm::dbgs() << "Domain Printer Starts\n");

  scopStream << numOps << "\n";
  ctx.targetFunc.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (shouldPrintOperation(op)) {
      affine::StatementInfo info = ctx.statementInfoMap[op];

      // for each domain, remove the first column that indicates equality or
      // inequality, as this info is now managed by the flat linear constraints
      for (SmallVector<int64_t> &row : info.domain)
        row.erase(row.begin());

      int numRows = info.domain.size();
      // If there is no rows, the stmt is not inside any loops
      // in this case, the columns are:
      // number of symbols + 1 constant col
      int numCols = numRows == 0 ? numIntegerArgs + 1 : info.domain[0].size();

      scopStream << "\n1\n";

      FlatLinearConstraints cst(
          numRows, 0, numCols, numCols - numIntegerArgs - 1, numIntegerArgs, 0);
      for (SmallVector<int64_t> &row : info.domain)
        cst.addInequality(row);

      printConstraints(cst, scopStream);
      scopStream << "0 0 0\n";
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "Domain Printer ends\n");
  scopStream << "0\n\n";

  // print the scattering
  LLVM_DEBUG(llvm::dbgs() << "Scatter Printer starts\n");
  scopStream << numOps << "\n";
  bool scatDimKnown = false;
  ctx.targetFunc.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (shouldPrintOperation(op)) {
      affine::StatementInfo info = ctx.statementInfoMap[op];

      // for each scattering, remove the first column that indicates equality or
      // inequality, as this info is now managed by the flat linear constraints
      for (SmallVector<int64_t> &scatterRow : info.scattering) {
        scatterRow.erase(scatterRow.begin());
        if (!scatDimKnown)
          ctx.scatteringDimension++; // We hijack the function to compute the
                                     // scattering dimension size
      }
      scatDimKnown = true;

      FlatLinearConstraints cst(
          0, info.scattering.size(), info.scattering[0].size(),
          info.scattering.size(),
          info.scattering[0].size() - info.scattering.size() - 1, 0);
      // inserting the constants (ordering variables) into the scattering
      for (size_t i = 0; i < info.constantRows.size(); i++)
        // ordering starts at -1, nad then -2, -3, etc
        info.scattering[i * 2][info.scattering[i].size() - 1] =
            -1 * info.constantRows[i];
      for (SmallVector<int64_t> &scatterRow : info.scattering)
        cst.addEquality(scatterRow);

      printConstraints(cst, scopStream);
      scopStream << "\n";
    }
  });

  // We set the iteration variables
  scopStream << "1\n";
  for (size_t i = 0; i < ctx.scatteringDimension; ++i) {
    if (i)
      scopStream << " ";
    if (i % 2 == 0) {
      scopStream << "cst_row" << std::to_string(i / 2);
    } else {
      scopStream << "k"
                 << std::to_string(i /
                                   2); // since i is of the form 2k+1, i/2 = k
    }
  }
  scopStream << '\n';
  LLVM_DEBUG(llvm::dbgs() << "Scatter Printer ends\n");
  scopStream.close();
}

void applyParallelOps(clast_stmt *root, Node *node, size_t iteratorsCount,
                      SmallVector<size_t> &ssaToSCoPMap) {
  for (auto child : node->children) {
    applyParallelOps(root, child, iteratorsCount, ssaToSCoPMap);
  }
  if (node && (node->transformation) &&
      (node->transformation)->transformationType == Parallel) {
    LabelParallel *lP = static_cast<LabelParallel *>((node->transformation));
    std::vector<int> temp;
    std::vector<int> check;
    // extract all statements from statement info
    // I am assuming they are sorted
    for (size_t i = 0; i < (lP->affectedStatements).size(); i++) {
      temp.push_back((lP->affectedStatements)[i]);
    }
    for (size_t i = 0; i < temp.size(); ++i) {
      if (check.size() == 0 ||
          check[check.size() - 1] != (int)ssaToSCoPMap[temp[i]]) {
        check.push_back(ssaToSCoPMap[temp[i]]);
      }
    }
    int64_t parallelDim = node->getDepth();
    std::string iter("k");
    iter += std::to_string(parallelDim);
    int nloops, nstmts;
    struct clast_for **loops;
    int *stmts;
    ClastFilter filter = {iter.c_str(), check.data(), (int)check.size(),
                          subset};
    clast_filter(root, filter, &loops, &nloops, &stmts, &nstmts);
    for (int i = 0; i < nloops; i++) {
      std::string privateVars;
      for (size_t j = parallelDim; j < iteratorsCount; j++) {
        privateVars.push_back('k');
        privateVars += std::to_string(j);
        privateVars += ", ";
      }
      privateVars += "ubv, lbv";
      loops[i]->private_vars = strdup(privateVars.c_str());
      loops[i]->parallel = CLAST_PARALLEL_OMP;
    }
  }
}

void applyVectorizeOps(clast_stmt *root, Node *node, size_t iteratorsCount,
                       SmallVector<size_t> &ssaToSCoPMap) {
  for (auto child : node->children) {
    applyVectorizeOps(root, child, iteratorsCount, ssaToSCoPMap);
  }
  if (node && (node->transformation) &&
      (node->transformation)->transformationType == Vectorize) {
    LabelVectorize *lV = static_cast<LabelVectorize *>((node->transformation));
    std::vector<int> temp;
    std::vector<int> check;
    // extract all statements from statement info
    // I am assuming they are sorted
    for (size_t i = 0; i < (lV->affectedStatements).size(); i++) {
      temp.push_back((lV->affectedStatements)[i]);
    }
    for (size_t i = 0; i < temp.size(); ++i) {
      if (check.size() == 0 ||
          check[check.size() - 1] != (int)ssaToSCoPMap[temp[i]]) {
        check.push_back(ssaToSCoPMap[temp[i]]);
      }
    }
    int64_t vectorizeDim = node->getDepth();
    std::string iter("k");
    iter += std::to_string(vectorizeDim);
    int nloops, nstmts;
    struct clast_for **loops;
    int *stmts;
    ClastFilter filter = {iter.c_str(), check.data(), (int)check.size(),
                          subset};
    clast_filter(root, filter, &loops, &nloops, &stmts, &nstmts);
    for (int i = 0; i < nloops; i++) {
      std::string privateVars;
      for (size_t j = vectorizeDim; j < iteratorsCount; j++) {
        privateVars.push_back('k');
        privateVars += std::to_string(j);
        privateVars.push_back(',');
      }
      privateVars += " ubv, lbv";
      loops[i]->private_vars = strdup(privateVars.c_str());
      loops[i]->parallel = CLAST_PARALLEL_VEC;
    }
  }
}

void applyUnrollOps(clast_stmt *root, Node *node,
                    SmallVector<size_t> &ssaToSCoPMap) {
  for (auto *child : node->children) {
    applyUnrollOps(root, child, ssaToSCoPMap);
  }
  if (node && (node->transformation) &&
      (node->transformation)->transformationType == Unroll) {
    LabelUnroll *lL = static_cast<LabelUnroll *>((node->transformation));
    std::vector<int> temp;
    std::vector<int> check;
    // extract all statements from statement info
    // I am assuming they are sorted
    for (size_t i = 0; i < (lL->affectedStatements).size(); i++) {
      temp.push_back((lL->affectedStatements)[i]);
    }
    for (size_t i = 0; i < temp.size(); ++i) {
      if (check.size() == 0 ||
          check[check.size() - 1] != (int)ssaToSCoPMap[temp[i]]) {
        check.push_back(ssaToSCoPMap[temp[i]]);
      }
    }
    int64_t unrollDim = node->getDepth();
    std::string iter("k");
    iter += std::to_string(unrollDim);
    int nloops, nstmts;
    struct clast_for **loops;
    int *stmts;
    ClastFilter filter = {iter.c_str(), check.data(), (int)check.size(),
                          subset};
    clast_filter(root, filter, &loops, &nloops, &stmts, &nstmts);
    for (int i = 0; i < nloops; i++) {
      loops[i]->unroll_type = clast_unroll_and_jam;
      loops[i]->ufactor = lL->unrollFactor;
      clast_unroll_jam(&loops[i]->stmt);
    }
  }
}

void printLoopStructures(UnparserState &ctx) {
  // TODO: support custom cloog file
  FILE *cloogSrc = fopen(ctx.cloogFileName.c_str(), "r");

  CloogState *state = cloog_state_malloc();
  CloogOptions *options = cloog_options_malloc(state);
  
  int nstmt = ctx.statementCount;
  options->fs = (int *)malloc(nstmt * sizeof(int));
  options->ls = (int *)malloc(nstmt * sizeof(int));
  options->fs_ls_size = nstmt;
  options->strides = 1;
  for (auto &stmt : ctx.statementInfoMap) {
    int fs = -1, ls = -1;
    SmallVector<bool> &isIteratorTiled = stmt.second.isIteratorTiled;
    // find second point loop for fs, -1 if not tiled
    bool *tiledLoopPtr =
        std::find(isIteratorTiled.begin(), isIteratorTiled.end(), true);
    if (tiledLoopPtr != isIteratorTiled.end()) {
      while (tiledLoopPtr != isIteratorTiled.end() && *tiledLoopPtr)
        tiledLoopPtr++;
      fs = tiledLoopPtr - isIteratorTiled.begin() + 1 + stmt.second.isFusedInto;
    }

    // count total number of loops for ls
    ls = isIteratorTiled.size() +
         std::count(isIteratorTiled.begin(), isIteratorTiled.end(), true) +
         stmt.second.isFusedInto * 2;

    // multiply the results by 2 since we interleave schedule rows and constant
    // rows
    if (fs != -1 && ls != -1) {
      options->fs[ctx.ssaToSCoPMap[stmt.second.opId] - 1] = fs * 2;
      options->ls[ctx.ssaToSCoPMap[stmt.second.opId] - 1] = ls * 2;
    } else {
      options->fs[ctx.ssaToSCoPMap[stmt.second.opId] - 1] = 1;
      options->ls[ctx.ssaToSCoPMap[stmt.second.opId] - 1] = -1;
    }
  }

  CloogInput *input = cloog_input_read(cloogSrc, options);

  fclose(cloogSrc);

  struct clast_stmt *root = cloog_clast_create_from_input(input, options);

  // Apply the parallelized, vectorized and unrolled loops
  applyParallelOps(root, ctx.tgRoot, ctx.scatteringDimension / 2,
                   ctx.ssaToSCoPMap);
  applyVectorizeOps(root, ctx.tgRoot, ctx.scatteringDimension / 2,
                    ctx.ssaToSCoPMap);
  applyUnrollOps(root, ctx.tgRoot, ctx.ssaToSCoPMap);

  std::string iterDecl = "int ";
  for (size_t i = 0; i < ctx.scatteringDimension; ++i) {
    // we only want iterators for odd values of i
    if (i % 2) {
      int iter = i / 2;
      if (iter)
        iterDecl += ", ";
      iterDecl += "k" + std::to_string(iter);
    }
  }
  ctx.varDefs += iterDecl + ";\n";

  // We need to get the output from cloog into a string
  // We could do this with fmemopen, but we don't know how
  // large cloog's output will be
  FILE *cOutput = fopen(ctx.codegenFileName.c_str(), "w");
  clast_pprint(cOutput, root, 0, options);
  fclose(cOutput);

  std::ifstream file(ctx.codegenFileName);
  std::stringstream buffer;
  buffer << file.rdbuf();
  ctx.loopStructure = buffer.str();

  cloog_clast_free(root);
  cloog_options_free(options);
  cloog_state_free(state);
}
/*
Unparser passes
*/

// This pass prints the include headers and the pre-defined macros
void headersAndMacrosPass(UnparserState &ctx) {
  llvm::raw_string_ostream code(ctx.headers);

  // Add standard includes
  code << "#include <math.h>\n";
  code << "#include <stdbool.h>\n";
  code << "#include <string.h>\n";
  code << "#include <stdint.h>\n";

  std::string benchmarkName = ctx.targetFunc.getSymName().str();
  ctx.benchmarkName = benchmarkName;

  bool flag = false;
  for (auto kernName : predefinedKernels) {
    if (kernName.first == benchmarkName) {
      flag = true;
    }
  }

  ctx.polybenchCodegen &= flag;

  // There are alternate semantics for polybench code generation
  if (ctx.polybenchCodegen) {
    std::string benchmarkName = ctx.targetFunc.getSymName().str();
    ctx.benchmarkName = benchmarkName;
    code << "#include <polybench.h>\n";
    code << "#include \"" << benchmarkName << ".h\"\n";
  }

  // we use a few custom macros which make code gen easier
  code << "#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))\n";
  code << "#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))\n";
  code << "#define max(x,y)    ((x) > (y) ? (x) : (y))\n";
  code << "#define min(x,y)    ((x) < (y) ? (x) : (y))\n";
  // these last two are for nusinov
  code << "#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0) \n";
  code << "#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2) \n";
  code << "\n";
}

// This pass prints the function signature
void functionSignaturePass(UnparserState &ctx) {
  llvm::raw_string_ostream code(ctx.signature);

  if (ctx.polybenchCodegen) {
    code << "\n\n" << predefinedKernels[ctx.benchmarkName];
    return;
  }

  code << "\n\n";

  // TODO: support return types :pensive-wobble:
  std::string retType = "void ";

  code << retType; // space included in function return value
  code << ctx.benchmarkName << '(';

  int argCounter = 1; // this is so the first argument is arg1
  mlir::MutableArrayRef<BlockArgument> args =
      ctx.targetFunc.getRegion().getArguments();

  bool start = true;

  for (mlir::Value argument : args) {
    if (start) {
      start = false;
    } else {
      code << ", ";
    }

    // Get the type of the argument and the name of the variable
    mlir::Type tpe = argument.getType();
    std::string varName = std::string("arg") + std::to_string(argCounter++);

    if (mlir::MemRefType mrtpe = dyn_cast_or_null<mlir::MemRefType>(tpe)) {
      // This case handles writing the variable name when it is a pointer
      // type
      mlir::Type elemTpe = mrtpe.getElementType();
      llvm::ArrayRef<int64_t> arrayShape = mrtpe.getShape();

      if (mlir::ShapedType::isDynamicShape(arrayShape)) {
        // This corresponds to the ? field in LLVM's memrefs,
        // we interprate that as DATA_TYPE *arg, and we also insist
        // that strides are passed in for array access.
        code << llvmTypeToCType(elemTpe) << "*" << varName;

        for (size_t i = 0; i < arrayShape.size(); ++i) {
          code << ", " << "int " << varName << "_stride_" << i;
        }
      } else {
        // This corresponds to an array type where the size
        // is known at compile time. IE int[3][2][4]
        code << llvmTypeToCType(elemTpe) << varName;
        for (int i : arrayShape) {
          code << "[" << i << "]";
        }
      }
    } else {
      // This case handles concrete types like int or float
      bool forLoopBound = false;
      if (tpe.isIndex()) {
        for (mlir::OpOperand &operand : argument.getUses()) {
          if (isa<AffineForOp>(operand.getOwner())) {
            forLoopBound = true;
            break;
          }
        }
      }
      if (forLoopBound) {
        // if the type is an index which is used
        // as the bounds of a for-loop. It must be
        // an int.
        code << "int " << varName;
      } else {
        code << llvmTypeToCType(tpe) << varName;
      }
    }
  }
  // remove the last comma
  code << ") ";
}

// This pass populates the valueToName map with arguments and loop variables
void populateVariablesPass(UnparserState &ctx) {
  ctx.varDefs += "int lbp, ubp, lbv, ubv;\n";

  // Add arguments to valueToName map
  int argCounter = 1;
  for (mlir::Value argument : ctx.targetFunc.getRegion().getArguments()) {
    ctx.valueToName[argument] =
        std::string("arg") + std::to_string(argCounter++);
  }

  // Walk all for-loops to get a list of iteration variables
  std::string baseIVName("i");
  ctx.targetFunc.walk<WalkOrder::PreOrder>([&](AffineForOp loop) {
    // so that iteration variables will be named i0, i1, i2, etc
    Operation *op = loop.getOperation();
    int depth = mlir::affine::getNestingDepth(op);

    ctx.valueToName[loop.getInductionVar()] =
        baseIVName + std::to_string(depth);
  });
}

// This pass determines the definitions of every statement
void statementDefinitionPass(UnparserState &ctx) {
  // We use a sort of messed up definition "static
  // control part". Static control parts are separated
  // by load and alloca ops
  int currentSCoP = 1;
  ctx.targetFunc.walk<WalkOrder::PreOrder>([&](Operation *op) {
    // we don't consider control flow primitives in mlir
    if (isControlFlow(op))
      return;

    ctx.ssaToSCoPMap.push_back(currentSCoP);

    affine::StatementInfo info = ctx.statementInfoMap[op];

    int numTiledDims = std::count(info.isIteratorTiled.begin(),
                                  info.isIteratorTiled.end(), true);
    int numLoops = info.isIteratorTiled.size() + numTiledDims +
                   (numTiledDims == 0 ? 0 : info.isFusedInto);

    // We match on the operand to produce the corresponding C code
    if (memref::AllocaOp alloca = dyn_cast<memref::AllocaOp>(op)) {
      // TODO: Ensure that this support for alloca matches Kevin's new local
      // variable feature
      // TODO: Ensure that this support dynamicly sized memrefs

      mlir::Value val = op->getResult(0);

      // For alloca, we want to add a variable to the SSA map, but we
      // don't want to declare it globally (for parallelization reasons)
      // This is why we don't use declget

      std::string varName;

      if (ctx.valueToName.find(val) == ctx.valueToName.end()) {
        varName = std::string("var") + std::to_string(ctx.varCount++);
        ctx.valueToName[val] = varName;
      } else {
        varName = ctx.valueToName[val];
      }
      std::string dtype =
          ctx.polybenchCodegen
              ? std::string("DATA_TYPE ")
              : llvmTypeToCType(alloca.getType().getElementType());
      std::string statementBody = dtype + varName;

      ArrayRef<int64_t> sizes = alloca.getMemref().getType().getShape();

      for (long long dimension : sizes) {
        statementBody += "[" + std::to_string(dimension) + "]";
      }
      if (sizes.size() == 1 && sizes[0] == 1)
        statementBody += "; " + varName + "[0] = 0";
      else
        statementBody += "; memset(" + varName + ", 0, sizeof " + varName + ")";

      ctx.defineStatement(statementBody, numLoops);

    } else if (AffineStoreOp store = dyn_cast<AffineStoreOp>(op)) {
      std::string data = ctx.valueToName[store.getValue()];
      std::string location = ctx.declget(store.getMemref());

      SmallVector<std::pair<Value, int>> surroundingLoopIVs =
          getNumTileLoops(ctx.statementInfoMap[op].isIteratorTiled, op,
                          ctx.statementInfoMap[op].isFusedInto);

      std::string index = parseAffineMap(op, surroundingLoopIVs, ctx);

      ctx.defineStatement(location + index + " = " + data, numLoops);
    } else if (memref::StoreOp store = dyn_cast<memref::StoreOp>(op)) {
      TypedValue<MemRefType> memref = store.getMemref();
      std::string data = ctx.valueToName[store.getValue()];
      std::string location = ctx.declget(memref);

      bool isDynamicMemref =
          ShapedType::isDynamicShape(memref.getType().getShape());
      OperandRange indicies = store.getIndices();
      SmallVector<std::string> parsedIndex(indicies.size());

      for (size_t i = 0; i < indicies.size(); ++i)
        parsedIndex[i] = ctx.valueToName[indicies[i]];

      std::string index = assembleMemRefIndex(isDynamicMemref, indicies.size(),
                                              location, parsedIndex);

      ctx.defineStatement(location + index + " = " + data, numLoops);
    } else if (memref::LoadOp load = dyn_cast<memref::LoadOp>(op)) {
      TypedValue<MemRefType> memref = load.getMemref();
      std::string location = ctx.declget(memref);

      bool isDynamicMemref =
          ShapedType::isDynamicShape(memref.getType().getShape());
      OperandRange indicies = load.getIndices();
      SmallVector<std::string> parsedIndex(indicies.size());

      for (size_t i = 0; i < indicies.size(); ++i)
        parsedIndex[i] = ctx.valueToName[indicies[i]];

      std::string index = assembleMemRefIndex(isDynamicMemref, indicies.size(),
                                              location, parsedIndex);

      ctx.valueToName[op->getResult(0)] = location + index;
    } else if (arith::IndexCastOp cast = dyn_cast<arith::IndexCastOp>(op)) {
      ctx.valueToName[op->getResult(0)] = ctx.valueToName[cast.getOperand()];
    } else if (arith::IndexCastUIOp cast = dyn_cast<arith::IndexCastUIOp>(op)) {
      ctx.valueToName[op->getResult(0)] = ctx.valueToName[cast.getOperand()];
    } else if (AffineLoadOp load = dyn_cast<AffineLoadOp>(op)) {
      std::string location = ctx.valueToName[load.getMemref()];

      SmallVector<std::pair<Value, int>> surroundingLoopIVs =
          getNumTileLoops(ctx.statementInfoMap[op].isIteratorTiled, op,
                          ctx.statementInfoMap[op].isFusedInto);

      std::string index = parseAffineMap(op, surroundingLoopIVs, ctx);

      ctx.valueToName[op->getResult(0)] = location + index;
    } else if (arith::AddFOp binOp = dyn_cast<arith::AddFOp>(op)) {
      std::string lhs = ctx.valueToName[binOp.getLhs()];
      std::string rhs = ctx.valueToName[binOp.getRhs()];
      ctx.valueToName[op->getResult(0)] =
          std::string("(") + lhs + " + " + rhs + ")";
    } else if (arith::AddIOp binOp = dyn_cast<arith::AddIOp>(op)) {
      std::string lhs = ctx.valueToName[binOp.getLhs()];
      std::string rhs = ctx.valueToName[binOp.getRhs()];
      ctx.valueToName[op->getResult(0)] =
          std::string("(") + lhs + " + " + rhs + ")";
    } else if (arith::SubFOp binOp = dyn_cast<arith::SubFOp>(op)) {
      std::string lhs = ctx.valueToName[binOp.getLhs()];
      std::string rhs = ctx.valueToName[binOp.getRhs()];
      ctx.valueToName[op->getResult(0)] =
          std::string("(") + lhs + " - " + rhs + ")";
    } else if (arith::SubIOp binOp = dyn_cast<arith::SubIOp>(op)) {
      std::string lhs = ctx.valueToName[binOp.getLhs()];
      std::string rhs = ctx.valueToName[binOp.getRhs()];
      ctx.valueToName[op->getResult(0)] =
          std::string("(") + lhs + " - " + rhs + ")";
    } else if (arith::MulFOp binOp = dyn_cast<arith::MulFOp>(op)) {
      std::string lhs = ctx.valueToName[binOp.getLhs()];
      std::string rhs = ctx.valueToName[binOp.getRhs()];
      ctx.valueToName[op->getResult(0)] =
          std::string("(") + lhs + " * " + rhs + ")";
    } else if (arith::MulIOp binOp = dyn_cast<arith::MulIOp>(op)) {
      std::string lhs = ctx.valueToName[binOp.getLhs()];
      std::string rhs = ctx.valueToName[binOp.getRhs()];
      ctx.valueToName[op->getResult(0)] =
          std::string("(") + lhs + " * " + rhs + ")";
    } else if (arith::DivFOp binOp = dyn_cast<arith::DivFOp>(op)) {
      std::string lhs = ctx.valueToName[binOp.getLhs()];
      std::string rhs = ctx.valueToName[binOp.getRhs()];
      ctx.valueToName[op->getResult(0)] =
          std::string("(") + lhs + " / " + rhs + ")";
    } else if (arith::DivSIOp binOp = dyn_cast<arith::DivSIOp>(op)) {
      std::string lhs = ctx.valueToName[binOp.getLhs()];
      std::string rhs = ctx.valueToName[binOp.getRhs()];
      ctx.valueToName[op->getResult(0)] =
          std::string("(") + lhs + " / " + rhs + ")";
    } else if (arith::NegFOp unaryOp = dyn_cast<arith::NegFOp>(op)) {
      std::string oper = ctx.valueToName[unaryOp.getOperand()];
      ctx.valueToName[op->getResult(0)] = std::string("-(") + oper + ")";
    } else if (math::SqrtOp unaryOp = dyn_cast<math::SqrtOp>(op)) {
      std::string oper = ctx.valueToName[unaryOp.getOperand()];
      ctx.valueToName[op->getResult(0)] = std::string("sqrt(") + oper + ")";
    } else if (math::ExpOp unaryOp = dyn_cast<math::ExpOp>(op)) {
      std::string oper = ctx.valueToName[unaryOp.getOperand()];
      ctx.valueToName[op->getResult(0)] = std::string("exp(") + oper + ")";
    } else if (math::PowFOp powOp = dyn_cast<math::PowFOp>(op)) {
      std::string base = ctx.valueToName[powOp.getOperand(0)];
      std::string power = ctx.valueToName[powOp.getOperand(1)];
      ctx.valueToName[op->getResult(0)] =
          std::string("pow(") + base + ", " + power + ")";
    } else if (arith::SelectOp selOp = dyn_cast<arith::SelectOp>(op)) {
      std::string condition = ctx.valueToName[selOp.getCondition()];
      std::string truePart = ctx.valueToName[selOp.getTrueValue()];
      std::string falsePart = ctx.valueToName[selOp.getFalseValue()];
      ctx.valueToName[op->getResult(0)] = std::string("(") + condition + " ? " +
                                          truePart + " : " + falsePart + ")";
    } else if (arith::CmpIOp cmpOp = dyn_cast<arith::CmpIOp>(op)) {
      std::string leftHandSide = ctx.valueToName[cmpOp.getLhs()];
      std::string rightHandSide = ctx.valueToName[cmpOp.getRhs()];

      std::string predicate;

      switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        predicate = " == ";
        break;
      case arith::CmpIPredicate::ne:
        predicate = " != ";
        break;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        predicate = " < ";
        break;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        predicate = " <= ";
        break;
        break;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        predicate = " > ";
        break;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        predicate = " >= ";
        break;
      default:
        llvm_unreachable("Unknown arith CmpIOp predicate.");
      }
      ctx.valueToName[op->getResult(0)] =
          std::string("(") + leftHandSide + predicate + rightHandSide + ")";
    } else if (arith::CmpFOp cmpOp = dyn_cast<arith::CmpFOp>(op)) {
      std::string leftHandSide = ctx.valueToName[cmpOp.getLhs()];
      std::string rightHandSide = ctx.valueToName[cmpOp.getRhs()];

      std::string predicate;

      switch (cmpOp.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
      case arith::CmpFPredicate::UEQ:
        predicate = " == ";
        break;
      case arith::CmpFPredicate::ONE:
      case arith::CmpFPredicate::UNE:
        predicate = " != ";
        break;
      case arith::CmpFPredicate::OLT:
      case arith::CmpFPredicate::ULT:
        predicate = " < ";
        break;
      case arith::CmpFPredicate::OLE:
      case arith::CmpFPredicate::ULE:
        predicate = " <= ";
        break;
        break;
      case arith::CmpFPredicate::OGT:
      case arith::CmpFPredicate::UGT:
        predicate = " > ";
        break;
      case arith::CmpFPredicate::OGE:
      case arith::CmpFPredicate::UGE:
        predicate = " >= ";
        break;
      default:
        llvm_unreachable("Unknown arith CmpFOp predicate.");
      }
      ctx.valueToName[op->getResult(0)] =
          std::string("(") + leftHandSide + predicate + rightHandSide + ")";
    } else if (arith::ConstantOp constOp = dyn_cast<arith::ConstantOp>(op)) {
      if (FloatAttr f = dyn_cast<FloatAttr>(constOp.getValue())) {
        ctx.valueToName[op->getResult(0)] =
            std::to_string(f.getValueAsDouble()) + "f";
      } else if (IntegerAttr i = dyn_cast<IntegerAttr>(constOp.getValue())) {
        ctx.valueToName[op->getResult(0)] = std::to_string(i.getInt());
      }
    } else if (func::CallOp funcall = dyn_cast<func::CallOp>(op)) {
      // TODO: In the future, it may be wise to save the result of the function
      // call
      // TODO: to a variable, so that we can avoid calling it repeatedly
      std::string funcName = funcall.getCallee().str();

      std::string output = funcName + "(";

      bool sep = false;
      for (mlir::Value funcArg : funcall.getOperands()) {
        if (sep) {
          output += ", ";
        } else {
          sep = true;
        }
        output += ctx.valueToName[funcArg];
      }
      output += ")";

      ctx.valueToName[op->getResult(0)] = output;
    } else if (arith::SIToFPOp castOp = dyn_cast<arith::SIToFPOp>(op)) {
      ctx.valueToName[op->getResult(0)] = ctx.valueToName[castOp.getIn()];
    } else if (arith::UIToFPOp castOp = dyn_cast<arith::UIToFPOp>(op)) {
      ctx.valueToName[op->getResult(0)] = ctx.valueToName[castOp.getIn()];
    } else if (memref::ViewOp view = dyn_cast<memref::ViewOp>(op)) {
      std::string elemType = llvmTypeToCType(view.getType().getElementType());
      std::string source = ctx.valueToName[view.getSource()];

      ctx.valueToName[op->getResult(0)] = "(" + elemType + "*)(" + source + ")";
    } else if (arith::ExtFOp ext = dyn_cast<arith::ExtFOp>(op)) {
      std::string tpe1 = llvmTypeToCType(ext.getOut().getType());
      std::string source = ctx.valueToName[ext.getIn()];

      ctx.valueToName[op->getResult(0)] = "(" + tpe1 + ")(" + source + ")";
    } else if (arith::ExtUIOp ext = dyn_cast<arith::ExtUIOp>(op)) {
      std::string tpe1 = llvmTypeToCType(ext.getOut().getType());
      std::string source = ctx.valueToName[ext.getIn()];

      ctx.valueToName[op->getResult(0)] = "(" + tpe1 + ")(" + source + ")";
    } else if (arith::ExtSIOp ext = dyn_cast<arith::ExtSIOp>(op)) {
      std::string tpe1 = llvmTypeToCType(ext.getOut().getType());
      std::string source = ctx.valueToName[ext.getIn()];

      ctx.valueToName[op->getResult(0)] = "(" + tpe1 + ")(" + source + ")";
    } else if (arith::TruncFOp trun = dyn_cast<arith::TruncFOp>(op)) {
      std::string tpe1 = llvmTypeToCType(trun.getOut().getType());
      std::string source = ctx.valueToName[trun.getIn()];

      ctx.valueToName[op->getResult(0)] = "(" + tpe1 + ")(" + source + ")";
    } else if (arith::TruncIOp trun = dyn_cast<arith::TruncIOp>(op)) {
      std::string tpe1 = llvmTypeToCType(trun.getOut().getType());
      std::string source = ctx.valueToName[trun.getIn()];

      ctx.valueToName[op->getResult(0)] = "(" + tpe1 + ")(" + source + ")";
    } else {
      op->dump();
      llvm_unreachable("unsupported mlir operation.");
    }

    if (shouldPrintOperation(op))
      ++currentSCoP;
  });
}

// This pass uses Cloog to print out all of the loop structures
void loopStructuresPass(UnparserState &ctx) {
  printOpenScopToFile(ctx);
  printLoopStructures(ctx);
}

void PolyTool::generateOptimizedCode() {
  // Build the context (we do this here as we need
  // polytool fields)

  codeGenOutputFile =
      (codeGenOutputFile == "")
          ? ("/tmp/cloog_c_out." + std::to_string(getpid()) + ".c")
          : codeGenOutputFile;

  UnparserState ctx(targetFunc, statementInfoMap, tgRoot, codeGenOutputFile,
                    polybenchCodeGen);

  std::cout << "OpenScop file: " << ctx.cloogFileName << "\n";
  std::cout << "Generated C file: " << codeGenOutputFile << "\n";

  headersAndMacrosPass(ctx);
  functionSignaturePass(ctx);
  populateVariablesPass(ctx);
  statementDefinitionPass(ctx);
  loopStructuresPass(ctx);

  std::cout << "Statement Count: " << ctx.statementCount << "\n";

  std::error_code ec;
  llvm::raw_fd_ostream result(llvm::StringRef(codeGenOutputFile), ec);

  result << ctx.headers << "\n"
         << ctx.statementDefs << ctx.signature << " {\n"
         << ctx.varDefs << ctx.loopStructure << "\n}\n";
}
