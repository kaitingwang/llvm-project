//#pragma scop
//  for (k = 0; k < _PB_N; k++) {
//    for (i = 0; i < _PB_N; i++)
//      for (j = 0; j < _PB_N; j++)
//        path[i][j] = path[i][j] < path[i][k] + path[k][j]
//                         ? path[i][j]
//                         : path[i][k] + path[k][j];
//  }
//#pragma endscop

module attributes {llvm.data_layout = "", llvm.target_triple = ""}  {
  func.func private @floyd_warshall(%arg0: i32, %arg1: memref<5600x5600xi32>) {
    %0 = arith.index_cast %arg0 : i32 to index // 0
    affine.for %arg2 = 0 to %0 { // 1
      affine.for %arg3 = 0 to %0 { // 1 0
        affine.for %arg4 = 0 to %0 { // 1 0 0
          %1 = affine.load %arg1[%arg3, %arg4] : memref<5600x5600xi32> // 1 0 0 0
          %2 = affine.load %arg1[%arg3, %arg2] : memref<5600x5600xi32> // 1 0 0 1
          %3 = affine.load %arg1[%arg2, %arg4] : memref<5600x5600xi32> // 1 0 0 2
          %4 = arith.addi %2, %3 : i32 // 1 0 0 3
          %5 = arith.cmpi "slt", %1, %4 : i32 // 1 0 0 4
          %6 = arith.select %5, %1, %4 : i32 // 1 0 0 5
          affine.store %6, %arg1[%arg3, %arg4] : memref<5600x5600xi32> // 1 0 0 6
        }
      } {tile}
    } 
    return
  }
  // for k ->                   1
  //     for iT ->              1 0
  //         for jT ->          1 0 0
  //             for ii ->      1 0 0 0
  //                 for jj ->  1 0 0 0 0
  //                      S1 -> 1 0 0 0 0 0
  //                      S2 -> 1 0 0 0 0 1
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %1 = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %22, %23, %24, %25 = transform.validator.tile %1 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.parallel %22 {force = true}  : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}
