module {
  func.func private @Twomm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: f64, %arg5: f64, %arg6: memref<?x1800xf64>, %arg7: memref<?x2200xf64>, %arg8: memref<?x1800xf64>, %arg9: memref<?x2400xf64>, %arg10: memref<?x2400xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    affine.for %arg11 = 0 to %3 {
      affine.for %arg12 = 0 to %0 {
        affine.store %cst, %arg6[%arg11, %arg12] {set = 0}: memref<?x1800xf64>
        affine.for %arg13 = 0 to %1 {
          %4 = affine.load %arg7[%arg11, %arg13] {set = 1}: memref<?x2200xf64>
          %5 = arith.mulf %arg4, %4 {set = 1}: f64
          %6 = affine.load %arg8[%arg13, %arg12] {set = 1}: memref<?x1800xf64>
          %7 = arith.mulf %5, %6 {set = 1}: f64
          %8 = affine.load %arg6[%arg11, %arg12] {set = 1}: memref<?x1800xf64>
          %9 = arith.addf %8, %7 {set = 1}: f64
          affine.store %9, %arg6[%arg11, %arg12] {set = 1}: memref<?x1800xf64>
        }
      }
    } {distribute1}
    affine.for %arg11 = 0 to %3 {
      affine.for %arg12 = 0 to %2 {
        %4 = affine.load %arg10[%arg11, %arg12] {set = 0}: memref<?x2400xf64>
        %5 = arith.mulf %4, %arg5 {set = 0}: f64
        affine.store %5, %arg10[%arg11, %arg12] {set = 0}: memref<?x2400xf64>
        affine.for %arg13 = 0 to %0 {
          %6 = affine.load %arg6[%arg11, %arg13] {set = 1}: memref<?x1800xf64>
          %7 = affine.load %arg9[%arg13, %arg12] {set = 1}: memref<?x2400xf64>
          %8 = arith.mulf %6, %7 {set = 1}: f64
          %9 = affine.load %arg10[%arg11, %arg12] {set = 1}: memref<?x2400xf64>
          %10 = arith.addf %9, %8 {set = 1}: f64
          affine.store %10, %arg10[%arg11, %arg12] {set = 1}: memref<?x2400xf64>
        }
      }
    } {distribute2}
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %23 = transform.structured.match ops{["affine.for"]} attributes {distribute1} in %arg1 : (!transform.any_op) -> !transform.any_op
    %24, %25 = transform.validator.distribute %23 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %26, %27, %28, %29, %30, %31 = transform.validator.tile %25 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.reorder %30 and %31 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %32 = transform.validator.parallel %26 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

    %13 = transform.structured.match ops{["affine.for"]} attributes {distribute2} in %arg1 : (!transform.any_op) -> !transform.any_op
    %14, %15 = transform.validator.distribute %13 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %16, %17, %18, %19, %20, %21 = transform.validator.tile %15 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.reorder %20 and %21 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %22 = transform.validator.parallel %16 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}
