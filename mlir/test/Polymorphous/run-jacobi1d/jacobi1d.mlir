// RUN: mlir-opt --correct-schedule --split-input-file %s
#map0 = affine_map<()[s0] -> (s0 - 1)>
module @"jacobi" attributes {llvm.data_layout = "", llvm.target_triple = ""} {
  func.func @jacobi1d(%T: index, %N: index, %a: memref<?xf32>, %b: memref<?xf32>) {
    affine.for %i = 0 to %T {
      affine.for %j = 1 to #map0()[%N] {
        %const = arith.constant 0.33333 : f32
        %1 = affine.load %a[%j - 1] : memref<?xf32>
        %2 = affine.load %a[%j] : memref<?xf32>
        %3 = affine.load %a[%j + 1] : memref<?xf32>
        %4 = arith.addf %1, %2 : f32
        %5 = arith.addf %4, %3 : f32
        %6 = arith.mulf %5, %const : f32
        affine.store %6, %b[%j] : memref<?xf32>
      } {fuse_1}
      affine.for %j = 1 to #map0()[%N] {
        %const = arith.constant 0.33333 : f32
        %1 = affine.load %b[%j - 1] : memref<?xf32>
        %2 = affine.load %b[%j] : memref<?xf32>
        %3 = affine.load %b[%j + 1] : memref<?xf32>
        %4 = arith.addf %1, %2 : f32
        %5 = arith.addf %4, %3 : f32
        %6 = arith.mulf %5, %const : f32
        affine.store %6, %a[%j] : memref<?xf32>
      } {fuse_2}
    } {tile}
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["affine.for"]} attributes{fuse_1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %1 = transform.structured.match ops{["affine.for"]} attributes{fuse_2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %2 = transform.validator.fuse %0 with %1 at 1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">)
    %3 = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4:4 = transform.validator.tile %3 { tile_sizes = [32, 32] } : (!transform.any_op) -> (!transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.parallel %4#1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %4#3 {factor = 8} : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}
