//===- AffineTransformOps.h - Affine transformation ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
namespace affine {
class AffineForOp;
} // namespace affine
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Schedule/TransformOps/ValidatorTransformOps.h.inc"
namespace mlir {

class DialectRegistry;

namespace validator {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace validator
} // namespace mlir