//   for (k = 0; k < _PB_N; k++) {
//     nrm = SCALAR_VAL(0.0);
//     for (i = 0; i < _PB_M; i++)
//       nrm += A[i][k] * A[i][k];
//     R[k][k] = SQRT_FUN(nrm);
//     for (i = 0; i < _PB_M; i++)
//       Q[i][k] = A[i][k] / R[k][k];
//     for (j = k + 1; j < _PB_N; j++) {
//       R[k][j] = SCALAR_VAL(0.0);
//       for (i = 0; i < _PB_M; i++)
//         R[k][j] += Q[i][k] * A[i][j];
//       for (i = 0; i < _PB_M; i++)
//         A[i][j] = A[i][j] - Q[i][k] * R[k][j];
//     }
//   }

#map = affine_map<(d0) -> (d0 + 1)>
module @"grandschmit" attributes {} {
  func.func @gramschmidt(%0: index, %2: index, %arg2: memref<2000x2600xf32>, %arg3: memref<2600x2600xf32>, %arg4: memref<2000x2600xf32>) {
    %1 = memref.alloca() : memref<1xf32>
    %cst = arith.constant 0.000000e+00 : f32 // 0
    affine.for %arg5 = 0 to %2 { // 1
      affine.store %cst, %1[0] : memref<1xf32> // 1 0
      affine.for %arg6 = 0 to %0 { // 1 1
        %5 = affine.load %arg2[%arg6, %arg5] : memref<2000x2600xf32>
        %6 = arith.mulf %5, %5 : f32
        %7 = affine.load %1[0] : memref<1xf32>
        %8 = arith.addf %7, %6 : f32
        affine.store %8, %1[0] : memref<1xf32>
      }
      %3 = affine.load %1[0] : memref<1xf32> // 1 2
      %4 = math.sqrt %3 : f32 // 1 3
      affine.store %4, %arg3[%arg5, %arg5] : memref<2600x2600xf32> // 1 4
      affine.for %arg6 = 0 to %0 { // 1 5
        %5 = affine.load %arg2[%arg6, %arg5] : memref<2000x2600xf32>
        %51 = affine.load %arg3[%arg5, %arg5] : memref<2600x2600xf32>
        %6 = arith.divf %5, %51 : f32
        affine.store %6, %arg4[%arg6, %arg5] : memref<2000x2600xf32>
      }
      affine.for %arg6 = #map(%arg5) to %2 { // 1 7
        affine.store %cst, %arg3[%arg5, %arg6] {set = 0, fuse_into} : memref<2600x2600xf32> // 1 6 0

        affine.for %arg7 = 0 to %0 { // 1 7 0
          %5 = affine.load %arg4[%arg7, %arg5] {set = 1}: memref<2000x2600xf32>
          %6 = affine.load %arg2[%arg7, %arg6] {set = 1}: memref<2000x2600xf32>
          %7 = arith.mulf %5, %6 {set = 1}: f32
          %8 = affine.load %arg3[%arg5, %arg6] {set = 1}: memref<2600x2600xf32>
          %9 = arith.addf %8, %7 {set = 1}: f32
          affine.store %9, %arg3[%arg5, %arg6] {set = 1}: memref<2600x2600xf32>
        } {fuse_1}
        affine.for %arg7 = 0 to %0 {
          %5 = affine.load %arg2[%arg7, %arg6] {set = 1}: memref<2000x2600xf32>
          %6 = affine.load %arg4[%arg7, %arg5] {set = 1}: memref<2000x2600xf32>
          %7 = affine.load %arg3[%arg5, %arg6] {set = 1}: memref<2600x2600xf32>
          %8 = arith.mulf %6, %7 {set = 1}: f32
          %9 = arith.subf %5, %8 {set = 1}: f32
          affine.store %9, %arg2[%arg7, %arg6] {set = 1}: memref<2000x2600xf32>
        } {fuse_2}
      } {tile, distribute, parallel}
    }
     return
   }
   transform.sequence failures(propagate) {
   ^bb1(%arg1: !transform.any_op):
    %2 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> !transform.any_op
    %output:2 = transform.validator.distribute %2 : (!transform.any_op) -> (!transform.any_op, !transform.op<"affine.for">)

    //iT  jT   ii   jj   iT  ii   jj -> This is an example of partial tiling
    %10, %11, %12, %13, %14, %15, %16 = transform.validator.tile %output#1 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %r0, %r1 = transform.validator.reorder %15 and %16 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %r2, %r3 = transform.validator.reorder %12 and %13 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.unroll %r1 {factor = 8} : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %r3 {factor = 8} : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %parallel = transform.validator.parallel %10 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
   }
}
