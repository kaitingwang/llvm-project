module {
  func.func @doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<250x220x270xf64>, %arg4: memref<270x270xf64>, %arg5: memref<270xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %2 {
      affine.for %arg7 = 0 to %0 {
        affine.for %arg8 = 0 to %1 {
          affine.store %cst, %arg5[%arg8] {set = 0}: memref<270xf64>
          affine.for %arg9 = 0 to %1 {
            %3 = affine.load %arg3[%arg6, %arg7, %arg9] {set = 1}: memref<250x220x270xf64>
            %4 = affine.load %arg4[%arg9, %arg8] {set = 1}: memref<270x270xf64>
            %5 = arith.mulf %3, %4 {set = 1}: f64
            %6 = affine.load %arg5[%arg8] {set = 1}: memref<270xf64>
            %7 = arith.addf %6, %5 {set = 1}: f64
            affine.store %7, %arg5[%arg8] {set = 1}: memref<270xf64>
          }
        } {distribute}
        affine.for %arg8 = 0 to %1 {
          %3 = affine.load %arg5[%arg8] : memref<270xf64>
          affine.store %3, %arg3[%arg6, %arg7, %arg8] : memref<250x220x270xf64>
        }
      }
    } 
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %2 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> !transform.any_op
    %11, %12 = transform.validator.distribute %2 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %19, %20, %21, %22 = transform.validator.tile %12 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    // transform.validator.reorder %21 and %22 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
  }
}

