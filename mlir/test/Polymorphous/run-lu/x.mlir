module {
  func.func private @lu(%arg0: index, %arg1: memref<40x40xf64>) attributes {PolyTool} {
    affine.for %arg2 = 0 to %arg0 {
      affine.for %arg3 = 0 to affine_map<(d0) -> (d0)>(%arg2) {
        affine.for %arg4 = 0 to affine_map<(d0) -> (d0)>(%arg3) {
          %3 = affine.load %arg1[%arg2, %arg4] : memref<40x40xf64>
          %4 = affine.load %arg1[%arg4, %arg3] : memref<40x40xf64>
          %5 = arith.mulf %3, %4 : f64
          %6 = affine.load %arg1[%arg2, %arg3] : memref<40x40xf64>
          %7 = arith.subf %6, %5 : f64
          affine.store %7, %arg1[%arg2, %arg3] : memref<40x40xf64>
        } {fuse_1}
        %0 = affine.load %arg1[%arg3, %arg3] {fuse_target1} : memref<40x40xf64>
        %1 = affine.load %arg1[%arg2, %arg3] {fuse_target2} : memref<40x40xf64>
        %2 = arith.divf %1, %0 {fuse_target3} : f64
        affine.store %2, %arg1[%arg2, %arg3] {fuse_target4} : memref<40x40xf64>
      } {fuse_4}
      affine.for %arg3 = affine_map<(d0) -> (d0)>(%arg2) to %arg0 {
        affine.for %arg4 = 0 to affine_map<(d0) -> (d0)>(%arg2) {
          %0 = affine.load %arg1[%arg2, %arg4] : memref<40x40xf64>
          %1 = affine.load %arg1[%arg4, %arg3] : memref<40x40xf64>
          %2 = arith.mulf %0, %1 : f64
          %3 = affine.load %arg1[%arg2, %arg3] : memref<40x40xf64>
          %4 = arith.subf %3, %2 : f64
          affine.store %4, %arg1[%arg2, %arg3] : memref<40x40xf64>
        }
      } {fuse_3}
    } {tile}
    return
  }
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["affine.for"]} attributes {fuse_1} in %arg0 : (!transform.any_op) -> !transform.op<"affine.for">
    %1 = transform.structured.match attributes {fuse_target1} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.validator.fuse_into %1 into %0 : (!transform.op<"affine.for">, !transform.any_op) -> !transform.op<"affine.for">
    %3 = transform.structured.match attributes {fuse_target2} in %arg0 : (!transform.any_op) -> !transform.any_op
    %4 = transform.validator.fuse_into %3 into %2 : (!transform.op<"affine.for">, !transform.any_op) -> !transform.op<"affine.for">
    %5 = transform.structured.match attributes {fuse_target3} in %arg0 : (!transform.any_op) -> !transform.any_op
    %6 = transform.validator.fuse_into %5 into %4 : (!transform.op<"affine.for">, !transform.any_op) -> !transform.op<"affine.for">
    %7 = transform.structured.match attributes {fuse_target4} in %arg0 : (!transform.any_op) -> !transform.any_op
    %8 = transform.validator.fuse_into %7 into %6 : (!transform.op<"affine.for">, !transform.any_op) -> !transform.op<"affine.for">
    %9 = transform.structured.match ops{["affine.for"]} attributes {fuse_3} in %arg0 : (!transform.any_op) -> !transform.op<"affine.for">
    %10 = transform.structured.match ops{["affine.for"]} attributes {fuse_4} in %arg0 : (!transform.any_op) -> !transform.op<"affine.for">
    %11 = transform.validator.fuse %10 with %9 at 2 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> !transform.op<"affine.for">
    %12 = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg0 : (!transform.any_op) -> !transform.op<"affine.for">
    %13:6 = transform.validator.tile %12 {tile_sizes = [32, 32, 32]} : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.any_op, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %transformed, %results = transform.validator.reorder %13#4 and %13#5 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %14 = transform.validator.parallel %13#1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}