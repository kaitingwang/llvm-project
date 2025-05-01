//===- PolyToolTransformationUtils.h - PolyTool transforms ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares IR modifying transformations of polytool
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_POLYTOOLTRANSFORMATIONS_H
#define MLIR_DIALECT_AFFINE_POLYTOOLTRANSFORMATIONS_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace schedule {

memref::AllocaOp expandAlloca(memref::AllocaOp alloca,
                              const SmallVector<int64_t> &sizes);

} // namespace schedule
} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_POLYTOOLTRANSFORMATIONS_H