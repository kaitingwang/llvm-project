// RUN: mlir-opt --correct-schedule --split-input-file %s
//
//  for (i = 0; i < _PB_N; i++) {
//    for (j = 0; j < i; j++) {
//      for (k = 0; k < j; k++) {
//        A[i][j] -= A[i][k] * A[k][j];
//      }
//      for (k = j; k < j+1;j++) {
//        A[i][j] /= A[j][j];
//      }
//    }
//    for (j = i; j < _PB_N; j++) {
//      for (k = 0; k < i; k++) {
//        A[i][j] -= A[i][k] * A[k][j];
//      }
//    }
//  }

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0+1)>
module {
  func.func private @lu(%0: index, %arg1: memref<40x40xf64>) {
    affine.for %arg2 = 0 to %0 { // 0
      affine.for %arg3 = 0 to #map(%arg2) { // 0 0
        affine.for %arg4 = 0 to #map(%arg3) { // 0 0 0
          %4 = affine.load %arg1[%arg2, %arg4] : memref<40x40xf64> // 0 0 0 0
          %5 = affine.load %arg1[%arg4, %arg3] : memref<40x40xf64> // 0 0 0 1
          %6 = arith.mulf %4, %5 : f64                             // 0 0 0 2
          %7 = affine.load %arg1[%arg2, %arg3] : memref<40x40xf64> // 0 0 0 3
          %8 = arith.subf %7, %6 : f64                             // 0 0 0 4
          affine.store %8, %arg1[%arg2, %arg3] : memref<40x40xf64> // 0 0 0 5
        } {fuse_1}
        %1 = affine.load %arg1[%arg3, %arg3] {fuse_target1}: memref<40x40xf64>  // 0 0 0 6
        %2 = affine.load %arg1[%arg2, %arg3] {fuse_target2}: memref<40x40xf64>  // 0 0 0 7
        %3 = arith.divf %2, %1 {fuse_target3}: f64                              // 0 0 0 8 
        affine.store %3, %arg1[%arg2, %arg3] {fuse_target4}: memref<40x40xf64>  // 0 0 0 9
      } {fuse_4}
      affine.for %arg3 = #map(%arg2) to %0 {
        affine.for %arg4 = 0 to #map(%arg2) {
          %1 = affine.load %arg1[%arg2, %arg4] : memref<40x40xf64> // 0 0 0 10
          %2 = affine.load %arg1[%arg4, %arg3] : memref<40x40xf64> // 0 0 0 11
          %3 = arith.mulf %1, %2 : f64                             // 0 0 0 12
          %4 = affine.load %arg1[%arg2, %arg3] : memref<40x40xf64> // 0 0 0 13
          %5 = arith.subf %4, %3 : f64                             // 0 0 0 14
          affine.store %5, %arg1[%arg2, %arg3] : memref<40x40xf64> // 0 0 0 15
        }
      } {fuse_3}
    } {tile}
    return
  }
  transform.sequence failures(propagate) {
    ^bb1(%arg1: !transform.any_op):

      %1 = transform.structured.match ops{["affine.for"]} attributes {fuse_1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %2 = transform.structured.match attributes {fuse_target1} in %arg1 : (!transform.any_op) -> !transform.any_op
      %5 = transform.validator.fuse_into %2 into %1 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)

      %6 = transform.structured.match attributes {fuse_target2} in %arg1 : (!transform.any_op) -> !transform.any_op
      %7 = transform.validator.fuse_into %6 into %5 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)

      %8 = transform.structured.match attributes {fuse_target3} in %arg1 : (!transform.any_op) -> !transform.any_op
      %9 = transform.validator.fuse_into %8 into %7 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)

      %10 = transform.structured.match attributes {fuse_target4} in %arg1 : (!transform.any_op) -> !transform.any_op
      %11 = transform.validator.fuse_into %10 into %9 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)

      %3 = transform.structured.match ops{["affine.for"]} attributes {fuse_3} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %4 = transform.structured.match ops{["affine.for"]} attributes {fuse_4} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %f2 = transform.validator.fuse %4 with %3 at 2 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">)

      %tile = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %12, %13, %14, %15, %16, %17 = transform.validator.tile %tile { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.any_op, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
      transform.validator.reorder %16 and %17 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)

      transform.validator.parallel %13 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}
