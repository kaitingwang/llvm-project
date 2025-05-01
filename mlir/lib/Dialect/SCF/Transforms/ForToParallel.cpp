//===- ForToParallel.cpp - scf.for to scf.parallel loop conversion --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms SCF.ForOp's into SCF.ParallelOp's.
//
// !!! Please note this pass performs no legality checks whatsoever !!!
// !!! and is intended to be used for a demo purpose only. !!! 
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SCFFORTOPARALLELLOOP
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace llvm;
using namespace mlir;
using scf::ForOp;
using scf::ParallelOp;
using scf::ReduceOp;

namespace {
struct ForLoopLoweringPattern : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto parallelLoop = rewriter.create<ParallelOp>(forOp.getLoc(), forOp.getLowerBound(),
                                         forOp.getUpperBound(), forOp.getStep());
    // Erase original terminator since parallelOp needs reduceOp to terminate
    rewriter.eraseOp(forOp.getBody()->getTerminator());

    // Inline for-loop body operations into 'parallel' body.
    for (auto &arg : llvm::make_early_inc_range(*forOp.getBody())) 
      arg.moveBefore(parallelLoop.getBody()->getTerminator());

    for (const auto &barg : enumerate(forOp.getBody(0)->getArguments()))
      rewriter.replaceAllUsesWith(barg.value(),
                                  parallelLoop.getBody()->getArgument(barg.index()));

    for (const auto &arg : llvm::enumerate(forOp.getResults()))
      rewriter.replaceAllUsesWith(arg.value(),
                                  parallelLoop.getResult(arg.index() + 1));
    rewriter.eraseOp(forOp);
    return success();
  }
};
struct ForToParallelLoop : public impl::SCFForToParallelLoopBase<ForToParallelLoop> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForLoopLoweringPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(parentOp, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createForToParallelLoopPass() {
  return std::make_unique<ForToParallelLoop>();
}
