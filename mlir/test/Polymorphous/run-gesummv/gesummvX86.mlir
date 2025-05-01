module {
  func.func private @gesummv(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<2800x2800xf64>, %arg4: memref<2800x2800xf64>, %arg5: memref<2800xf64>, %arg6: memref<2800xf64>, %arg7: memref<2800xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg8 = 0 to %0 {
      affine.store %cst, %arg5[%arg8] {set = 0}: memref<2800xf64>
      affine.store %cst, %arg7[%arg8] {set = 0}: memref<2800xf64>
      affine.for %arg9 = 0 to %0 {
        %6 = affine.load %arg3[%arg8, %arg9] {set = 1}: memref<2800x2800xf64>
        %7 = affine.load %arg6[%arg9] {set = 1}: memref<2800xf64>
        %8 = arith.mulf %6, %7 {set = 1}: f64
        %9 = affine.load %arg5[%arg8] {set = 1}: memref<2800xf64>
        %10 = arith.addf %8, %9 {set = 1}: f64
        affine.store %10, %arg5[%arg8] {set = 1}: memref<2800xf64>
        %11 = affine.load %arg4[%arg8, %arg9] {set = 2}: memref<2800x2800xf64>
        %15 = affine.load %arg6[%arg9] {set = 2}: memref<2800xf64>
        %12 = arith.mulf %11, %15 {set = 2}: f64
        %13 = affine.load %arg7[%arg8] {set = 2}: memref<2800xf64>
        %14 = arith.addf %12, %13 {set = 2}: f64
        affine.store %14, %arg7[%arg8] {set = 2}: memref<2800xf64>
      }
      %1 = affine.load %arg5[%arg8] {set = 3}: memref<2800xf64>
      %2 = arith.mulf %arg1, %1 {set = 3}: f64
      %3 = affine.load %arg7[%arg8] {set = 3}: memref<2800xf64>
      %4 = arith.mulf %arg2, %3 {set = 3}: f64
      %5 = arith.addf %2, %4 {set = 3}: f64
      affine.store %5, %arg7[%arg8] {set = 3}: memref<2800xf64>
    } {distribute}
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    // Old Optimizations
    // %2 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> (!transform.op<"affine.for">)
    // // //%11, %12, %13 = transform.validator.distribute %2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.op<"affine.for">)
    // // //%15, %16, %17, %18 = transform.validator.tile %12 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.any_op, !transform.any_op)
    // %21 = transform.validator.parallel %2 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

    // New Optimizations
    %2 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> (!transform.any_op)
    %3, %4, %5, %6 = transform.validator.distribute %2 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %7, %8, %9, %10 = transform.validator.tile %4 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.any_op)
    %11, %12, %13, %14 = transform.validator.tile %5 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.any_op)
    transform.validator.parallel %3 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %4 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %7 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %11 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %13 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %9 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}

