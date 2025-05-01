#map = affine_map<(d0) -> (d0)>
module {
  func.func private @symm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<2000x2600xf64>, %arg5: memref<2000x2000xf64>, %arg6: memref<2000x2600xf64>) {
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = memref.alloca() {schedule.expand = [1, 1205]}: memref<1xf64>
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %2 {
      affine.for %arg8 = 0 to %0 {
        affine.store %cst_0, %1[0] {_set = 0}: memref<1xf64>
        affine.for %arg9 = 0 to #map(%arg7) {
          %14 = affine.load %arg6[%arg7, %arg8] {_set = 1}: memref<2000x2600xf64>
          %15 = arith.mulf %arg2, %14 {_set = 1}: f64
          %16 = affine.load %arg5[%arg7, %arg9] {_set = 1}: memref<2000x2000xf64>
          %17 = arith.mulf %15, %16 {_set = 1}: f64
          %18 = affine.load %arg4[%arg9, %arg8] {_set = 1}: memref<2000x2600xf64>
          %19 = arith.addf %18, %17 {_set = 1}: f64
          affine.store %19, %arg4[%arg9, %arg8] {_set = 1}: memref<2000x2600xf64>
          %161 = affine.load %arg5[%arg7, %arg9] {_set = 2}: memref<2000x2000xf64>
          %20 = affine.load %arg6[%arg9, %arg8] {_set = 2}: memref<2000x2600xf64>
          %21 = arith.mulf %20, %161 {_set = 2}: f64
          %22 = affine.load %1[0] {_set = 2}: memref<1xf64>
          %23 = arith.addf %22, %21 {_set = 2}: f64
          affine.store %23, %1[0] {_set = 2}: memref<1xf64>
        }
        %4 = affine.load %arg4[%arg7, %arg8] {_set = 2}: memref<2000x2600xf64>
        %5 = arith.mulf %arg3, %4 {_set = 2}: f64
        %6 = affine.load %arg6[%arg7, %arg8] {_set = 2}: memref<2000x2600xf64>
        %7 = arith.mulf %arg2, %6 {_set = 2}: f64
        %8 = affine.load %arg5[%arg7, %arg7] {_set = 2}: memref<2000x2000xf64>
        %9 = arith.mulf %7, %8 {_set = 2}: f64
        %10 = arith.addf %5, %9 {_set = 2}: f64
        %11 = affine.load %1[0] {_set = 2}: memref<1xf64>
        %12 = arith.mulf %arg2, %11 {_set = 2}: f64
        %13 = arith.addf %10, %12 {_set = 2}: f64
        affine.store %13, %arg4[%arg7, %arg8] {_set = 2}: memref<2000x2600xf64>
      } {parallel}
    } 
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %14 = transform.structured.match ops{["affine.for"]} attributes {parallel} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %21 = transform.validator.parallel %14 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}

