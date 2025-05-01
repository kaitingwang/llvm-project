// RUN: mlir-opt --correct-schedule --split-input-file %s
//
//   for (t = 0; t <= _PB_TSTEPS - 1; t++)
//     for (i = 1; i <= _PB_N - 2; i++)
//       for (j = 1; j <= _PB_N - 2; j++)
//         A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] +
//                    A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] +
//                    A[i + 1][j] + A[i + 1][j + 1]) /
//                   SCALAR_VAL(9.0);

#map0 = affine_map<()[s0] -> (s0 - 1)>
module @"seidal" attributes {llvm.data_layout = "e-i1:8:32-i8:8:32-i16:16:32-i64:64-f16:16:32-v16:16-v32:32-n64-S64", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  func.func @seidal(%_PB_TSTEPS: index, %_PB_N: index, %A: memref<?x?xf32>) {
    affine.for %t = 0 to %_PB_TSTEPS { 
      affine.for %i = 1 to #map0()[%_PB_N] { 
        affine.for %j = 1 to #map0()[%_PB_N] { 
          %const = arith.constant  9.0 : f32
          %1 = affine.load %A[%i - 1, %j - 1]  : memref<?x?xf32>
          %2 = affine.load %A[%i - 1, %j]  : memref<?x?xf32>
          %3 = affine.load %A[%i - 1, %j + 1]  : memref<?x?xf32>
          %4 = affine.load %A[%i, %j - 1]  : memref<?x?xf32>
          %5 = affine.load %A[%i, %j]  : memref<?x?xf32>
          %6 = affine.load %A[%i, %j + 1]  : memref<?x?xf32>
          %7 = affine.load %A[%i + 1, %j - 1]  : memref<?x?xf32>
          %8 = affine.load %A[%i + 1, %j]  : memref<?x?xf32>
          %9 = affine.load %A[%i + 1, %j + 1]  : memref<?x?xf32>

          %a1 = arith.addf %1, %2 : f32
          %a2 = arith.addf %a1, %3 : f32
          %a3 = arith.addf %a2, %4 : f32
          %a4 = arith.addf %a3, %5 : f32
          %a5 = arith.addf %a4, %6 : f32
          %a6 = arith.addf %a5, %7 : f32
          %a7 = arith.addf %a6, %8 : f32
          %a8 = arith.addf %a7, %9 : f32
          %res = arith.divf %a8, %const : f32

          affine.store %res, %A[%i, %j]  : memref<?x?xf32>
        } 
      } {parallel}
    } {tile}
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    //%3 = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg1 : (!transform.any_op) -> !transform.any_op
    %3 = transform.structured.match ops{["affine.for"]} attributes {parallel} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %parallel = transform.validator.parallel %3 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    //%4:6 = transform.validator.tile %3 { tile_sizes = [32, 32, 32] } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  }
}
