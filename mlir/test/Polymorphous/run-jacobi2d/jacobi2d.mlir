// RUN: mlir-opt --correct-schedule --split-input-file %s
//
// for (t = 0; t < _PB_TSTEPS; t++)
//   {
//     for (i = 1; i < _PB_N - 1; i++)
//        for (j = 1; j < _PB_N - 1; j++)
//           B[i, j] = SCALAR_VAL(0.2) * (A[i, j] + A[i, j-1] + A[i, 1+j] + A[1+i, j] + A[i-1, j]);
//     for (i = 1; i < _PB_N - 1; i++)
//        for (j = 1; j < _PB_N - 1; j++)
//           A[i, j] = SCALAR_VAL(0.2) * (B[i, j] + B[i, j-1] + B[i, 1+j] + B[1+i, j] + B[i-1, j]);
//   }

#map0 = affine_map<()[s0] -> (s0 - 1)>
module @"jacobi" attributes {llvm.data_layout = "", llvm.target_triple = ""} {
  func.func @jacobi2d(%T: index, %N: index, %a: memref<?x?xf32>, %b: memref<?x?xf32>) {
    affine.for %t = 0 to %T { 
      affine.for %i = 1 to #map0()[%N] {
        affine.for %j = 1 to #map0()[%N] {
          %const = arith.constant  0.2 : f32
          %1 = affine.load %a[%i, %j - 1] : memref<?x?xf32>
          %2 = affine.load %a[%i, %j] : memref<?x?xf32>
          %3 = affine.load %a[%i, %j + 1] : memref<?x?xf32>
          %4 = affine.load %a[%i - 1, %j] : memref<?x?xf32>
          %5 = affine.load %a[%i + 1, %j] : memref<?x?xf32>
          %6 = arith.addf %1, %2 : f32
          %7 = arith.addf %6, %3 : f32
          %8 = arith.addf %7, %4 : f32
          %9 = arith.addf %8, %5 : f32
          %10 = arith.mulf %9, %const : f32
          affine.store %10, %b[%i, %j] : memref<?x?xf32>
        } 
      } {tile}
      affine.for %i = 1 to #map0()[%N] {
        affine.for %j = 1 to #map0()[%N] {
          %const = arith.constant  0.2 : f32
          %1 = affine.load %b[%i, %j - 1]: memref<?x?xf32>
          %2 = affine.load %b[%i, %j]: memref<?x?xf32>
          %3 = affine.load %b[%i, %j + 1]: memref<?x?xf32>
          %4 = affine.load %b[%i - 1, %j]: memref<?x?xf32>
          %5 = affine.load %b[%i + 1, %j]: memref<?x?xf32>
          %6 = arith.addf %1, %2: f32
          %7 = arith.addf %6, %3: f32
          %8 = arith.addf %7, %4: f32
          %9 = arith.addf %8, %5: f32
          %10 = arith.mulf %9, %const: f32
          affine.store %10, %a[%i, %j]: memref<?x?xf32>
        } 
      } {lastHope}
    } 
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %1 = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // %22, %23, %24, %25, %26, %27 = transform.validator.tile %1 { tile_sizes = [1, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %2 = transform.structured.match ops{["affine.for"]} attributes {lastHope} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %parallel1 = transform.validator.parallel %1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %parallel2 = transform.validator.parallel %2 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    // %32, %33, %34, %35, %36, %37 = transform.validator.tile %2 { tile_sizes = [1, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    // transform.validator.parallel %22  : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    // transform.validator.parallel %32  : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    // //%11, %12 = transform.validator.distribute %2 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.any_op)
    // // %0 = transform.structured.match ops{["affine.for"]} attributes{fuse_1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // // %1 = transform.structured.match ops{["affine.for"]} attributes{fuse_2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // // %2 = transform.validator.fuse %0 with %1 at 2 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">)

    // // %3 = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg1 : (!transform.any_op) -> !transform.any_op
    // // %4:6 = transform.validator.tile %3 { tile_sizes = [8, 32, 32] } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  }
}
