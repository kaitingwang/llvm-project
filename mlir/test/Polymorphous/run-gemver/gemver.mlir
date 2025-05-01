module {
  func.func private @gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<4000x4000xf64>, %arg4: memref<4000xf64>, %arg5: memref<4000xf64>, %arg6: memref<4000xf64>, %arg7: memref<4000xf64>, %arg8: memref<4000xf64>, %arg9: memref<4000xf64>, %arg10: memref<4000xf64>, %arg11: memref<4000xf64>) {
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg3[%arg12, %arg13] : memref<4000x4000xf64>
        %2 = affine.load %arg4[%arg12] : memref<4000xf64>
        %3 = affine.load %arg5[%arg13] : memref<4000xf64>
        %4 = arith.mulf %2, %3 : f64
        %5 = arith.addf %1, %4 : f64
        %6 = affine.load %arg6[%arg12] : memref<4000xf64>
        %7 = affine.load %arg7[%arg13] : memref<4000xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = arith.addf %5, %8 : f64
        affine.store %9, %arg3[%arg12, %arg13] : memref<4000x4000xf64>
      }
    } {tile1}
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg9[%arg12] : memref<4000xf64>
        %2 = affine.load %arg3[%arg13, %arg12] : memref<4000x4000xf64>
        %3 = arith.mulf %arg2, %2 : f64
        %4 = affine.load %arg10[%arg13] : memref<4000xf64>
        %5 = arith.mulf %3, %4 : f64
        %6 = arith.addf %1, %5 : f64
        affine.store %6, %arg9[%arg12] : memref<4000xf64>
      } 
    } {tile2}
    affine.for %arg12 = 0 to %0 {
      %1 = affine.load %arg9[%arg12] : memref<4000xf64>
      %2 = affine.load %arg11[%arg12] : memref<4000xf64>
      %3 = arith.addf %1, %2 : f64
      affine.store %3, %arg9[%arg12] : memref<4000xf64>
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg8[%arg12] : memref<4000xf64>
        %2 = affine.load %arg3[%arg12, %arg13] : memref<4000x4000xf64>
        %3 = arith.mulf %arg1, %2 : f64
        %4 = affine.load %arg9[%arg13] : memref<4000xf64>
        %5 = arith.mulf %3, %4 : f64
        %6 = arith.addf %1, %5 : f64
        affine.store %6, %arg8[%arg12] : memref<4000xf64>
      }
    } {tile3}
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %2 = transform.structured.match ops{["affine.for"]} attributes {tile1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %3, %4, %5, %6 = transform.validator.tile %2 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.any_op)
    %7 = transform.structured.match ops{["affine.for"]} attributes {tile2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %8, %9, %10, %11 = transform.validator.tile %7 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %20, %21 = transform.validator.reorder %10 and %11 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %12 = transform.structured.match ops{["affine.for"]} attributes {tile3} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %13, %14, %15, %16 = transform.validator.tile %12 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %17 = transform.validator.parallel %3 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %18 = transform.validator.parallel %8 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %19 = transform.validator.parallel %13 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %5 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %21 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %15 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}

