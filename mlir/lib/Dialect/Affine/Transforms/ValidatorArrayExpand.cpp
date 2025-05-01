//===- ValidatorArrayExpand.cpp - Pass to expand arrays -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/PolyToolUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Schedule/PolyToolTransformationUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_VALIDATORARRAYEXPAND
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "validator-array-expand"

using namespace mlir;
using namespace mlir::affine;

namespace {
struct ValidatorArrayExpandPass
    : public affine::impl::ValidatorArrayExpandBase<ValidatorArrayExpandPass> {

  void runOnOperation() override {
    constexpr StringRef expandAttrName("schedule.expand");
    getOperation()->walk([&](memref::AllocaOp alloca) {
      if (alloca->hasAttr(expandAttrName)) {
        Attribute attr = alloca->getAttr(expandAttrName);
        SmallVector<int64_t> expandSizes = extractFromI64ArrayAttr(attr);
        schedule::expandAlloca(alloca, expandSizes);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createValidatorArrayExpandPass() {
  return std::make_unique<ValidatorArrayExpandPass>();
}
