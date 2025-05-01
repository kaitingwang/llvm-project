#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func private @syr2k(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: f64, %arg4: memref<2600x2600xf64>, %arg5: memref<2600x2000xf64>, %arg6: memref<2600x2000xf64>) {
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg7 = 0 to %0 {
      affine.for %arg8 = 0 to #map(%arg7) {
        %2 = affine.load %arg4[%arg7, %arg8] {set = 0}: memref<2600x2600xf64>
        %3 = arith.mulf %2, %arg3 {set = 0}: f64
        affine.store %3, %arg4[%arg7, %arg8] {set = 0}: memref<2600x2600xf64>
      } 
      affine.for %arg8 = 0 to %1 {
        affine.for %arg9 = 0 to #map(%arg7) {
          %2 = affine.load %arg5[%arg9, %arg8] {set = 1}: memref<2600x2000xf64>
          %3 = arith.mulf %2, %arg2 {set = 1}: f64
          %4 = affine.load %arg6[%arg7, %arg8] {set = 1}: memref<2600x2000xf64>
          %5 = arith.mulf %3, %4 {set = 1}: f64
          %6 = affine.load %arg6[%arg9, %arg8] {set = 1}: memref<2600x2000xf64>
          %7 = arith.mulf %6, %arg2 {set = 1}: f64
          %8 = affine.load %arg5[%arg7, %arg8] {set = 1}: memref<2600x2000xf64>
          %9 = arith.mulf %7, %8 {set = 1}: f64
          %10 = arith.addf %5, %9 {set = 1}: f64
          %11 = affine.load %arg4[%arg7, %arg9] {set = 1}: memref<2600x2600xf64>
          %12 = arith.addf %11, %10 {set = 1}: f64
          affine.store %12, %arg4[%arg7, %arg9] {set = 1}: memref<2600x2600xf64>
        }
      }
    } {distribute}
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %13 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> !transform.any_op
    %14, %15 = transform.validator.distribute %13 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    //%23, %24, %25, %26 = transform.validator.tile %14 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %16, %17, %18, %19, %20, %21 = transform.validator.tile %15 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %30, %31 = transform.validator.reorder %20 and %21 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %22 = transform.validator.parallel %16 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    //%27 = transform.validator.parallel %23 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    //transform.validator.unroll %31 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}

