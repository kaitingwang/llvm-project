// RUN: mlir-opt --correct-schedule --split-input-file %s
module {
  func.func @mvt(%_PB_N:index, %x1: memref<?xf32>, %x2: memref<?xf32>, 
                        %y1: memref<?xf32>, %y2: memref<?xf32>, %A: memref<?x?xf32>) {        
      affine.for %i = 0 to %_PB_N {
          affine.for %j = 0 to %_PB_N {
              %1 = affine.load %A[%i, %j] : memref<?x?xf32>
              %2 = affine.load %y1[%j] : memref<?xf32>
              %3 = arith.mulf %1, %2 : f32
              %4 = affine.load %x1[%i] : memref<?xf32>
              %5 = arith.addf %4, %3 : f32
              affine.store %5, %x1[%i] : memref<?xf32>
          }
      } {tile1}
      affine.for %i = 0 to %_PB_N {
          affine.for %j = 0 to %_PB_N {
              %1 = affine.load %A[%j, %i] : memref<?x?xf32>
              %2 = affine.load %y2[%j] : memref<?xf32>
              %3 = arith.mulf %1, %2 : f32
              %4 = affine.load %x2[%i] : memref<?xf32>
              %5 = arith.addf %4, %3 : f32
              affine.store %5, %x2[%i] : memref<?xf32>
          } 
      } {tile2}
      return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    // Old Optimization
    // %0 = transform.structured.match ops{["affine.for"]} attributes {reorder = 0} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // %1 = transform.structured.match ops{["affine.for"]} attributes {reorder = 1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // //%2, %3 = transform.validator.reorder %0 and %1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    // %4 = transform.structured.match ops{["affine.for"]} attributes{fuse} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // %5 = transform.validator.fuse %1 with %4 at 2 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">)
    // %6:4 = transform.validator.tile %5 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    // %7, %8 = transform.validator.reorder %6#2 and %6#3 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    // %9 = transform.validator.parallel %6#0 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">


    ////// New Optimizations
    %0 = transform.structured.match ops{["affine.for"]} attributes {tile1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %1 = transform.structured.match ops{["affine.for"]} attributes {tile2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %3, %4, %5, %6 = transform.validator.tile %0 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %7, %8, %9, %10 = transform.validator.tile %1 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %11, %12 = transform.validator.reorder %9 and %10 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.parallel %7 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %3 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %5 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %12 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}
