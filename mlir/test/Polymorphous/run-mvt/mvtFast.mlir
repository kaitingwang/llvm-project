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
    %2 = transform.structured.match ops{["affine.for"]} attributes {tile1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %3, %4, %5, %6 = transform.validator.tile %2 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %parallel = transform.validator.parallel %3 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    
    %7 = transform.structured.match ops{["affine.for"]} attributes {tile2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %8, %9, %10, %11 = transform.validator.tile %7 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.reorder %10 and %11 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %parallel2 = transform.validator.parallel %8 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}
