// #pragma scop
//   /* E := A*B */
//   for (i = 0; i < _PB_NI; i++)
//     for (j = 0; j < _PB_NJ; j++) {
//       E[i][j] = SCALAR_VAL(0.0);
//       for (k = 0; k < _PB_NK; ++k)
//         E[i][j] += A[i][k] * B[k][j];
//     }
//   /* F := C*D */
//   for (i = 0; i < _PB_NJ; i++)
//     for (j = 0; j < _PB_NL; j++) {
//       F[i][j] = SCALAR_VAL(0.0);
//       for (k = 0; k < _PB_NM; ++k)
//         F[i][j] += C[i][k] * D[k][j];
//     }
//   /* G := E*F */
//   for (i = 0; i < _PB_NI; i++)
//     for (j = 0; j < _PB_NL; j++) {
//       G[i][j] = SCALAR_VAL(0.0);
//       for (k = 0; k < _PB_NJ; ++k)
//         G[i][j] += E[i][k] * F[k][j];
//     }
// #pragma endscop

module {
  func.func private @Threemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<1600x1800xf64>, %arg6: memref<1600x2000xf64>, %arg7: memref<2000x1800xf64>, %arg8: memref<1800x2200xf64>, %arg9: memref<1800x2400xf64>, %arg10: memref<2400x2200xf64>, %arg11: memref<1600x2200xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = arith.index_cast %arg4 : i32 to index
    %4 = arith.index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to %4 {
      affine.for %arg13 = 0 to %0 {
        affine.store %cst, %arg5[%arg12, %arg13] {set = 0}: memref<1600x1800xf64>
        affine.for %arg14 = 0 to %1 {
          %5 = affine.load %arg6[%arg12, %arg14] {set = 1}: memref<1600x2000xf64>
          %6 = affine.load %arg7[%arg14, %arg13] {set = 1}: memref<2000x1800xf64>
          %7 = arith.mulf %5, %6 {set = 1}: f64
          %8 = affine.load %arg5[%arg12, %arg13] {set = 1}: memref<1600x1800xf64>
          %9 = arith.addf %8, %7 {set = 1}: f64
          affine.store %9, %arg5[%arg12, %arg13] {set = 1}: memref<1600x1800xf64>
        }
      }
    } {distribute1}

    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %2 {
        affine.store %cst, %arg8[%arg12, %arg13] {set = 0}: memref<1800x2200xf64>
        affine.for %arg14 = 0 to %3 {
          %5 = affine.load %arg9[%arg12, %arg14] {set = 1}: memref<1800x2400xf64>
          %6 = affine.load %arg10[%arg14, %arg13] {set = 1}: memref<2400x2200xf64>
          %7 = arith.mulf %5, %6 {set = 1}: f64
          %8 = affine.load %arg8[%arg12, %arg13] {set = 1}: memref<1800x2200xf64>
          %9 = arith.addf %8, %7 {set = 1}: f64
          affine.store %9, %arg8[%arg12, %arg13] {set = 1}: memref<1800x2200xf64>
        }
      }
    } {distribute2}

    affine.for %arg12 = 0 to %4 {
      affine.for %arg13 = 0 to %2 {
        affine.store %cst, %arg11[%arg12, %arg13] {set = 0}: memref<1600x2200xf64>
        affine.for %arg14 = 0 to %0 {
          %5 = affine.load %arg5[%arg12, %arg14] {set = 1}: memref<1600x1800xf64>
          %6 = affine.load %arg8[%arg14, %arg13] {set = 1}: memref<1800x2200xf64>
          %7 = arith.mulf %5, %6 {set = 1}: f64
          %8 = affine.load %arg11[%arg12, %arg13] {set = 1}: memref<1600x2200xf64>
          %9 = arith.addf %8, %7 {set = 1}: f64
          affine.store %9, %arg11[%arg12, %arg13] {set = 1}: memref<1600x2200xf64>
        }
      }
    } {distribute3}
    return
  }
 transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %13 = transform.structured.match ops{["affine.for"]} attributes {distribute1} in %arg1 : (!transform.any_op) -> !transform.any_op
    %14, %15 = transform.validator.distribute %13 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %16, %17, %18, %19, %20, %21 = transform.validator.tile %15 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.reorder %20 and %21 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %22 = transform.validator.parallel %16 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

    %23 = transform.structured.match ops{["affine.for"]} attributes {distribute2} in %arg1 : (!transform.any_op) -> !transform.any_op
    %24, %25 = transform.validator.distribute %23 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %26, %27, %28, %29, %30, %31 = transform.validator.tile %25 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.reorder %30 and %31 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %32 = transform.validator.parallel %26 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

    %33 = transform.structured.match ops{["affine.for"]} attributes {distribute3} in %arg1 : (!transform.any_op) -> !transform.any_op
    %34, %35 = transform.validator.distribute %33 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %36, %37, %38, %39, %40, %41 = transform.validator.tile %35 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.reorder %40 and %41 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %42 = transform.validator.parallel %36 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  
  }
}

