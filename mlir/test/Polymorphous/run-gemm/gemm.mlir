//#pragma scop
//  for (i = 0; i < _PB_NI; i++) {
//    for (j = 0; j < _PB_NJ; j++)
//      C[i][j] *= beta;
//    for (k = 0; k < _PB_NK; k++) {
//      for (j = 0; j < _PB_NJ; j++)
//        C[i][j] += alpha * A[i][k] * B[k][j];
//    }
//  }
//#pragma endscop
module {
  func.func private @gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f32, %arg4: f32, %arg5: memref<?x?xf32>, %arg6: memref<?x?xf32>, %arg7: memref<?x?xf32>) {
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    affine.for %arg8 = 0 to %0 {
      affine.for %arg9 = 0 to %1 {
        %3 = affine.load %arg5[%arg8, %arg9] {set = 0}: memref<?x?xf32>
        %4 = arith.mulf %3, %arg4 {set = 0}: f32
        affine.store %4, %arg5[%arg8, %arg9] {set = 0}: memref<?x?xf32>
      }
      affine.for %arg9 = 0 to %2 {
        affine.for %arg10 = 0 to %1 {
          %3 = affine.load %arg6[%arg8, %arg9] {set = 1}: memref<?x?xf32>
          %4 = arith.mulf %arg3, %3 {set = 1}: f32
          %5 = affine.load %arg7[%arg9, %arg10] {set = 1}: memref<?x?xf32>
          %6 = arith.mulf %4, %5 {set = 1}: f32
          %7 = affine.load %arg5[%arg8, %arg10] {set = 1}: memref<?x?xf32>
          %8 = arith.addf %7, %6 {set = 1}: f32
          affine.store %8, %arg5[%arg8, %arg10] {set = 1}: memref<?x?xf32>
        } {vectorize}
      } 
    } {distribute}
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %2 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> !transform.any_op
    %output:2 = transform.validator.distribute %2 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    //%21, %22, %23, %24 = transform.validator.tile %output#0 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.any_op, !transform.any_op)
    %15, %16, %17, %18, %19, %20 = transform.validator.tile %output#1 { tile_sizes = [32, 26, 40] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.any_op, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.parallel %15 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

    // %vec = transform.structured.match ops{["affine.for"]} attributes {vectorize} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %vectorized = transform.validator.vectorize %20 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    // // transform.validator.unroll %vectorized { factor = 4 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}
