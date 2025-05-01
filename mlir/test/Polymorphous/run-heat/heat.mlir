// for (t = 1; t <= TSTEPS; t++) {
//     for (i = 1; i < _PB_N-1; i++) {
//         for (j = 1; j < _PB_N-1; j++) {
//             for (k = 1; k < _PB_N-1; k++) {
//                 B[i][j][k] =   0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
//                              + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
//                              + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
//                              + A[i][j][k];
//             }
//         }
//     }
//     for (i = 1; i < _PB_N-1; i++) {
//        for (j = 1; j < _PB_N-1; j++) {
//            for (k = 1; k < _PB_N-1; k++) {
//                A[i][j][k] =   0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
//                             + 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
//                             + 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
//                             + B[i][j][k];
//            }
//        }
//    }
// }

#map0 = affine_map<()[s0] -> (s0 - 1)>
module @"heat" attributes {llvm.data_layout = "e-i1:8:32-i8:8:32-i16:16:32-i64:64-f16:16:32-v16:16-v32:32-n64-S64", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @heat(%T: index, %N: index, %a: memref<?x?x?xf32>, %b: memref<?x?x?xf32>) {
    affine.for %t = 0 to %T { 
      affine.for %i = 1 to #map0()[%N] {
        affine.for %j = 1 to #map0()[%N] {
          affine.for %k = 1 to #map0()[%N] {
            %const1 = arith.constant  2.0 : f32
            %const2 = arith.constant  0.125 : f32
            %1 = affine.load %a[%i + 1, %j, %k]  : memref<?x?x?xf32>
            %2 = affine.load %a[%i, %j, %k]  : memref<?x?x?xf32>
            %3 = affine.load %a[%i - 1, %j, %k]  : memref<?x?x?xf32>
            %4 = affine.load %a[%i, %j + 1, %k]  : memref<?x?x?xf32>
            %5 = affine.load %a[%i, %j - 1, %k]  : memref<?x?x?xf32>
            %6 = affine.load %a[%i, %j, %k + 1]  : memref<?x?x?xf32>
            %7 = affine.load %a[%i, %j, %k - 1]  : memref<?x?x?xf32>
            %8 = arith.mulf %2, %const1  : f32
            %9 = arith.subf %1, %8  : f32
            %10 = arith.addf %9, %3  : f32
            %11 = arith.mulf %10, %const2 : f32

            %12 = arith.subf %4, %8  : f32
            %13 = arith.addf %12, %5  : f32
            %14 = arith.mulf %13, %const2 : f32

            %15 = arith.subf %6, %8  : f32
            %16 = arith.addf %15, %7  : f32
            %17 = arith.mulf %16, %const2 : f32

            %18 = arith.addf %11, %14 : f32
            %19 = arith.addf %18, %17 : f32
            %20 = arith.addf %19, %2 : f32
            affine.store %20, %b[%i, %j, %k]  : memref<?x?x?xf32>
          } 
        } 
      } {fuse_1, parallel1}
      affine.for %i = 1 to #map0()[%N] {
        affine.for %j = 1 to #map0()[%N] {
          affine.for %k = 1 to #map0()[%N] {
            %const1 = arith.constant  2.0 : f32
            %const2 = arith.constant  0.125 : f32
            %1 = affine.load %b[%i + 1, %j, %k]  : memref<?x?x?xf32>
            %2 = affine.load %b[%i, %j, %k]  : memref<?x?x?xf32>
            %3 = affine.load %b[%i - 1, %j, %k]  : memref<?x?x?xf32>
            %4 = affine.load %b[%i, %j + 1, %k]  : memref<?x?x?xf32>
            %5 = affine.load %b[%i, %j - 1, %k]  : memref<?x?x?xf32>
            %6 = affine.load %b[%i, %j, %k + 1]  : memref<?x?x?xf32>
            %7 = affine.load %b[%i, %j, %k - 1]  : memref<?x?x?xf32>
            %8 = arith.mulf %2, %const1  : f32
            %9 = arith.subf %1, %8  : f32
            %10 = arith.addf %9, %3  : f32
            %11 = arith.mulf %10, %const2 : f32

            %12 = arith.subf %4, %8  : f32
            %13 = arith.addf %12, %5  : f32
            %14 = arith.mulf %13, %const2 : f32

            %15 = arith.subf %6, %8  : f32
            %16 = arith.addf %15, %7  : f32
            %17 = arith.mulf %16, %const2 : f32

            %18 = arith.addf %11, %14 : f32
            %19 = arith.addf %18, %17 : f32
            %20 = arith.addf %19, %2 : f32
            affine.store %20, %a[%i, %j, %k]  : memref<?x?x?xf32>
          } 
        } 
      } {fuse_2, parallel2}
    } {tile}

    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match attributes{parallel1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %1 = transform.structured.match attributes{parallel2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // %0 = transform.structured.match ops{["affine.for"]} attributes{fuse_1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // %1 = transform.structured.match ops{["affine.for"]} attributes{fuse_2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // %2 = transform.validator.fuse %0 with %1 at 3 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">)

    transform.validator.parallel %0 : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">)
    transform.validator.parallel %1 : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">)
    // // transform.validator.parallel %2 : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">)

    // %3 = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg1 : (!transform.any_op) -> !transform.any_op
    // // %4:8 = transform.validator.tile %3 { tile_sizes = [32, 32, 32, 512] } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // // %4:8 = transform.validator.tile %3 { tile_sizes = [32, 32, 32, 32] } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // %4:8 = transform.validator.tile %3 { tile_sizes = [8, 8, 32, 32] } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  }

}
