module @jacobi attributes {llvm.data_layout = "", llvm.target_triple = ""} {
  func.func @jacobi1d(%arg0: index, %arg1: index, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {PolyTool} {
    %cst = arith.constant 3.333300e-01 : f32
    affine.for %arg4 = 0 to %arg0 {
      affine.for %arg5 = 1 to affine_map<()[s0] -> (s0 - 1)>()[%arg1] {
        %0 = affine.load %arg2[%arg5 - 1] : memref<?xf32>
        %1 = affine.load %arg2[%arg5] : memref<?xf32>
        %2 = affine.load %arg2[%arg5 + 1] : memref<?xf32>
        %3 = arith.addf %0, %1 : f32
        %4 = arith.addf %3, %2 : f32
        %5 = arith.mulf %4, %cst : f32
        affine.store %5, %arg3[%arg5] : memref<?xf32>
      } {fuse_1}
      affine.for %arg5 = 1 to affine_map<()[s0] -> (s0 - 1)>()[%arg1] {
        %0 = affine.load %arg3[%arg5 - 1] : memref<?xf32>
        %1 = affine.load %arg3[%arg5] : memref<?xf32>
        %2 = affine.load %arg3[%arg5 + 1] : memref<?xf32>
        %3 = arith.addf %0, %1 : f32
        %4 = arith.addf %3, %2 : f32
        %5 = arith.mulf %4, %cst : f32
        affine.store %5, %arg2[%arg5] : memref<?xf32>
      } {fuse_2}
    } {tile}
    return
  }
  transform.sequence  failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["affine.for"]} attributes {fuse_1} in %arg0 : (!transform.any_op) -> !transform.op<"affine.for">
    %1 = transform.structured.match ops{["affine.for"]} attributes {fuse_2} in %arg0 : (!transform.any_op) -> !transform.op<"affine.for">
    %2 = transform.validator.fuse %0 with %1 at 1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> !transform.op<"affine.for">
    %3 = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg0 : (!transform.any_op) -> !transform.any_op
    %4:4 = transform.validator.tile %3 {tile_sizes = [32, 32]} : (!transform.any_op) -> (!transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %5 = transform.validator.parallel %4#1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %6 = transform.validator.unroll %4#3 {factor = 8 : i64} : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}