// RUN: mlir-opt --correct-schedule --split-input-file %s
//
//  for (i = 0; i < _PB_N; i++) {
//     //j<i
//     for (j = 0; j < i; j++) {
//        for (k = 0; k < j; k++) {
//           A[i][j] -= A[i][k] * A[j][k];
//        }
//        A[i][j] /= A[j][j];
//     }
//     // i==j case
//     for (k = 0; k < i; k++) {
//        A[i][i] -= A[i][k] * A[i][k];
//     }
//     A[i][i] = SQRT_FUN(A[i][i]);
//  }

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0+1)>
module {
  func.func private @cholesky(%0: index, %arg1: memref<?x?xf32>) {
    affine.for %arg2 = 0 to %0 {
      affine.for %arg3 = 0 to #map(%arg2) {
        affine.for %arg4 = 0 to #map(%arg3) {
          %6 = affine.load %arg1[%arg2, %arg4] : memref<?x?xf32>
          %7 = affine.load %arg1[%arg3, %arg4] : memref<?x?xf32>
          %8 = arith.mulf %6, %7 : f32
          %9 = affine.load %arg1[%arg2, %arg3] : memref<?x?xf32>
          %10 = arith.subf %9, %8 : f32
          affine.store %10, %arg1[%arg2, %arg3] : memref<?x?xf32>
        }{fuse_1}
        %3 = affine.load %arg1[%arg3, %arg3] {fuse_target1}: memref<?x?xf32>
        %4 = affine.load %arg1[%arg2, %arg3] {fuse_target2}: memref<?x?xf32>
        %5 = arith.divf %4, %3 {fuse_target3}: f32
        affine.store %5, %arg1[%arg2, %arg3] {fuse_target4}: memref<?x?xf32>
      }{fuse_2}
      affine.for %arg4 = 0 to #map(%arg2) {
        %3 = affine.load %arg1[%arg2, %arg4] {fuse_target8, manual_set1}: memref<?x?xf32>
        %4 = arith.mulf %3, %3 {fuse_target9, manual_set2}: f32
        %5 = affine.load %arg1[%arg2, %arg2] {fuse_target10, manual_set3}: memref<?x?xf32>
        %6 = arith.subf %5, %4 {fuse_target11, manual_set4}: f32
        affine.store %6, %arg1[%arg2, %arg2] {fuse_target12, manual_set5}: memref<?x?xf32>
      }{fuse_3}
      %1 = affine.load %arg1[%arg2, %arg2]  {fuse_target5}: memref<?x?xf32>
      %2 = math.sqrt %1 {fuse_target6}: f32
      affine.store %2, %arg1[%arg2, %arg2] {fuse_target7}: memref<?x?xf32>
    }{tile}
    return
  }
  transform.sequence failures(propagate) {
    ^bb1(%arg1: !transform.any_op):

      // //move line 32 - 35 into line24
      %1 = transform.structured.match ops{["affine.for"]} attributes {fuse_1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %2 = transform.structured.match attributes {fuse_target1} in %arg1 : (!transform.any_op) -> !transform.any_op
      %5 = transform.validator.fuse_into %2 into %1 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)
      %6 = transform.structured.match attributes {fuse_target2} in %arg1 : (!transform.any_op) -> !transform.any_op
      %7 = transform.validator.fuse_into %6 into %5 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)
      %8 = transform.structured.match attributes {fuse_target3} in %arg1 : (!transform.any_op) -> !transform.any_op
      %9 = transform.validator.fuse_into %8 into %7 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)
      %10 = transform.structured.match attributes {fuse_target4} in %arg1 : (!transform.any_op) -> !transform.any_op
      %11 = transform.validator.fuse_into %10 into %9 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)

      // %12 = transform.structured.match ops{["affine.for"]} attributes {fuse_2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      // %13 = transform.structured.match ops{["affine.for"]} attributes {fuse_3} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">

      //fuse line 37 and line 23
      //%f2 = transform.validator.fuse %13 with %12 at 2 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">)
      
      //move line 38-42 into line 24
      %15 = transform.structured.match attributes {fuse_target8} in %arg1 : (!transform.any_op) -> !transform.any_op
      %16 = transform.validator.fuse_into %15 into %11 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)
      %17 = transform.structured.match attributes {fuse_target9} in %arg1 : (!transform.any_op) -> !transform.any_op
      %18 = transform.validator.fuse_into %17 into %16 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)
      %19 = transform.structured.match attributes {fuse_target10} in %arg1 : (!transform.any_op) -> !transform.any_op
      %20 = transform.validator.fuse_into %19 into %18 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)
      %21 = transform.structured.match attributes {fuse_target11} in %arg1 : (!transform.any_op) -> !transform.any_op
      %22 = transform.validator.fuse_into %21 into %20 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)
      %23 = transform.structured.match attributes {fuse_target12} in %arg1 : (!transform.any_op) -> !transform.any_op
      %24 = transform.validator.fuse_into %23 into %22 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)
      
      //move line 44-46 into line 24
      %25 = transform.structured.match attributes {fuse_target5} in %arg1 : (!transform.any_op) -> !transform.any_op
      %26 = transform.validator.fuse_into %25 into %24 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)
      %27 = transform.structured.match attributes {fuse_target6} in %arg1 : (!transform.any_op) -> !transform.any_op
      %28 = transform.validator.fuse_into %27 into %26 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)
      %29 = transform.structured.match attributes {fuse_target7} in %arg1 : (!transform.any_op) -> !transform.any_op
      %30 = transform.validator.fuse_into %29 into %28 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)

      %31 = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %32:6 = transform.validator.tile %31 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
      // transform.validator.unroll %32#5 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
      transform.validator.unroll %32#3 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
      transform.validator.unroll %32#4 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
      transform.validator.parallel %32#1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
      
      %ms1 = transform.structured.match attributes {manual_set1} in %arg1 : (!transform.any_op) -> !transform.any_op
      %ms2 = transform.structured.match attributes {manual_set2} in %arg1 : (!transform.any_op) -> !transform.any_op
      %ms3 = transform.structured.match attributes {manual_set3} in %arg1 : (!transform.any_op) -> !transform.any_op
      %ms4 = transform.structured.match attributes {manual_set4} in %arg1 : (!transform.any_op) -> !transform.any_op
      %ms5 = transform.structured.match attributes {manual_set5} in %arg1 : (!transform.any_op) -> !transform.any_op

      transform.validator.manual_set %ms1 {schedule = [[1,  1,   0,   0,   0], 
                                                       [0,  1,   0,   0,   0], 
                                                       [0,   0,  1,   0,   0], 
                                                       [0,   0,   0,  1,   0], 
                                                       [0,   0,   0,   0,  1], 
                                                       [0,   0,   0,  1,   0]]} : (!transform.any_op) -> !transform.any_op
      transform.validator.manual_set %ms2 {schedule = [[1,  1,   0,   0,   0], 
                                                       [0,  1,   0,   0,   0], 
                                                       [0,   0,  1,   0,   0], 
                                                       [0,   0,   0,  1,   0], 
                                                       [0,   0,   0,   0,  1], 
                                                       [0,   0,   0,  1,   0]]} : (!transform.any_op) -> !transform.any_op
      transform.validator.manual_set %ms3 {schedule = [[1,  1,   0,   0,   0], 
                                                       [0,  1,   0,   0,   0], 
                                                       [0,   0,  1,   0,   0], 
                                                       [0,   0,   0,  1,   0], 
                                                       [0,   0,   0,   0,  1], 
                                                       [0,   0,   0,  1,   0]]} : (!transform.any_op) -> !transform.any_op
      transform.validator.manual_set %ms4 {schedule = [[1,  1,   0,   0,   0], 
                                                       [0,  1,   0,   0,   0], 
                                                       [0,   0,  1,   0,   0], 
                                                       [0,   0,   0,  1,   0], 
                                                       [0,   0,   0,   0,  1], 
                                                       [0,   0,   0,  1,   0]]} : (!transform.any_op) -> !transform.any_op
      transform.validator.manual_set %ms5 {schedule = [[1,  1,   0,   0,   0], 
                                                       [0,  1,   0,   0,   0], 
                                                       [0,   0,  1,   0,   0], 
                                                       [0,   0,   0,  1,   0], 
                                                       [0,   0,   0,   0,  1], 
                                                       [0,   0,   0,  1,   0]]} : (!transform.any_op) -> !transform.any_op
  }
}
