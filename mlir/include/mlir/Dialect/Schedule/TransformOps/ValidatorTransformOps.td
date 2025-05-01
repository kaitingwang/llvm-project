//===- ValidatorTransformOps.td - Validator transformation ops ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------------===//

#ifndef VALIDATOR_TRANSFORM_OPS
#define VALIDATOR_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Transform_AffineForOp : Transform_ConcreteOpType<"affine.for">;

def ValidatorManualSetOp : Op<Transform_Dialect, "validator.manual_set", [
    FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Manually insert a schedule for the target statement.";

  let arguments = (ins TransformHandleTypeInterface:$target,
         ArrayAttr:$schedule);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = "$target attr-dict `:`"
    "functional-type(operands, results)";
}

def ValidatorUnrollOp : Op<Transform_Dialect, "validator.unroll", [
    FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Unroll loop through #pragma statements.";
  let description = [{Unroll the current loop.}];

  let arguments = (ins TransformHandleTypeInterface:$target,
         DefaultValuedAttr<I64Attr, "4">:$factor);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = "$target attr-dict `:`"
    "functional-type(operands, results)";
}

def ValidatorVectorizeOp : Op<Transform_Dialect, "validator.vectorize", [
    FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Vectorize loop.";
  let description = [{Vectorize the current loop.}];

  let arguments = (ins TransformHandleTypeInterface:$target,
      DefaultValuedOptionalAttr<BoolAttr, "false">:$force);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = "$target attr-dict `:`"
    "functional-type(operands, results)";
}

def ValidatorParallelOp : Op<Transform_Dialect, "validator.parallel", [
    FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Parallelize loop.";
  let description = [{Parallelize the current loop.}];

  let arguments = (ins TransformHandleTypeInterface:$target,
      DefaultValuedOptionalAttr<BoolAttr, "false">:$force);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = "$target attr-dict `:`"
    "functional-type(operands, results)";
}

def ValidatorDistributedParallelOp : Op<Transform_Dialect, "validator.distributed_parallel", [
    FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Distributed parallelize the loop via MPI.";
  let description = [{Distributed parallelize the current loop.}];

  let arguments = (ins TransformHandleTypeInterface:$target,
      DefaultValuedOptionalAttr<I64Attr, "1">:$nproc);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = "$target attr-dict `:`"
    "functional-type(operands, results)";
}

def ValidatorDistributeOp: Op<Transform_Dialect, "validator.distribute",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Distribute a loop into multiple loops.";
  let description = [{
    Assumes the statements in the loop are properly labeled
    so we know which ones should be in the first loop, which
    ones should be in the second loop, or third loop.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);

  let results = (outs Variadic<TransformHandleTypeInterface> : $loops);

  let hasCustomAssemblyFormat = 1;
}

def ValidatorReorderOp : Op<Transform_Dialect, "validator.reorder", [
    FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>
  ]> {
  let summary = "reorder operation";
  let description = [{
    This is a reorder operation.
  }];

  let arguments = (ins TransformHandleTypeInterface:$first_loop,
                       TransformHandleTypeInterface:$second_loop);

  let results = (outs TransformHandleTypeInterface:$transformed, 
                      Variadic<TransformHandleTypeInterface>:$results);

  let assemblyFormat = "$first_loop `and` $second_loop attr-dict  `:`" 
    "functional-type(operands, results)";
}

def ValidatorPartialReorderOp : Op<Transform_Dialect, "validator.partial_reorder", [
    FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>
  ]> {
  let summary = "partial reorder operation";
  let description = [{
    This is a partial reorder operation. Returns nothing
  }];

  let arguments = (ins TransformHandleTypeInterface:$first_loop,
                       TransformHandleTypeInterface:$second_loop);

  let assemblyFormat = "$first_loop `and` $second_loop attr-dict `:`"
    "functional-type(operands, results)";
}

def ValidatorFuseOp : Op<Transform_Dialect, "validator.fuse", [
    FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>
  ]> {
  let summary = "fuse 2 target loops if valid";
  let description = [{
    The operation fuses the 2 target loops at a specific depth when
    the 2 loops are perfectly nested at the depth.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target1,
                       TransformHandleTypeInterface:$target2,
                       I64Attr:$depth);

  let results = (outs TransformHandleTypeInterface:$fused);

  let assemblyFormat = "$target1 `with` $target2 `at` $depth attr-dict  `:`"
    "functional-type(operands, results)";
}

def ValidatorLoopTilingOp : Op<Transform_Dialect, "validator.tile", [
    FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>
 ]> {
  let summary = "Tiles the selected loops to the specified tile size";

  let description = [{Tiles any loops within the tileBegin and tileEnd attribute.
      The first loop to be tiled must have the tildBegin attribute and the last to be
      tiled will have the tileEnd attribute. If no tildEnd is specified, then all
      loops within the tileBegin loop will be tiled. Only perfectly nested looped will
      be tiled.}];

  let arguments =
     (ins TransformHandleTypeInterface:$target,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes);

  let results = (outs Variadic<TransformHandleTypeInterface> : $loops);

  let hasCustomAssemblyFormat = 1;
}

def ValidatorBlockReorderOp : Op<Transform_Dialect, "validator.block_reorder", [
    FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
    DeclareOpInterfaceMethods<TransformOpInterface>
 ]> {
  let summary = "Reorder blocks of code according to the permutation vector.";

  let description = [{Reorder Blocks of code at same scop level. 
    Limitation: currently only support nested for loop blocks at scope 0.}];

  let arguments =
     (ins Variadic<TransformHandleTypeInterface>: $inputs,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$permutation);

  let results = (outs Variadic<TransformHandleTypeInterface>: $outputs);

  let assemblyFormat = "$inputs attr-dict `:` functional-type(operands, results)";
}

def ValidatorFuseIntoOp: Op<Transform_Dialect, "validator.fuse_into",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "fuse a single operation into a loop operation";

  let arguments = (ins TransformHandleTypeInterface:$loop,
                       TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$fused);

  let assemblyFormat = "$target `into` $loop attr-dict  `:` functional-type(operands, results)";
}

def ValidatorSkewOp: Op<Transform_Dialect, "validator.skew",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Skew a outer in the direction of inner by amount.";
  let description = [{
    Skew a outer loop in the direction of inner loop by amount.
  }];

  let arguments = (ins TransformHandleTypeInterface:$outer,
                       TransformHandleTypeInterface:$inner,
                        DefaultValuedAttr<I64Attr, "4">:$amount);

  let results = (outs TransformHandleTypeInterface:$skewed, TransformHandleTypeInterface:$unchanged);

  let assemblyFormat = "$outer `with` $inner `by` $amount attr-dict  `:` functional-type(operands, results)";
}


#endif // Validator_TRANSFORM_OPS
