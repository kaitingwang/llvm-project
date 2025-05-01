//   for (i = 0; i < _PB_N; i++)
//   {
//       x[i] = b[i];
//       for (j = 0; j <i; j++)
//         x[i] -= L[i][j] * x[j];
//       x[i] = x[i] / L[i][i];
//   }

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0+1)>
module {
   func.func private @trisolv(%0: index, %arg1: memref<40x40xf64>, %arg2: memref<40xf64>, %arg3: memref<40xf64>) {
     affine.for %arg4 = 0 to %0 {
       %1 = affine.load %arg3[%arg4] {set=0} : memref<40xf64>
       affine.store %1, %arg2[%arg4] {set=0} : memref<40xf64>
       affine.for %arg5 = 0 to #map(%arg4) {
         %5 = affine.load %arg1[%arg4, %arg5] {set=1} : memref<40x40xf64>
         %6 = affine.load %arg2[%arg5] {set=1} : memref<40xf64>
         %7 = arith.mulf %5, %6 {set=1} : f64
         %8 = affine.load %arg2[%arg4] {set=1} : memref<40xf64>
         %9 = arith.subf %8, %7 {set=1} : f64
         affine.store %9, %arg2[%arg4] {set=1} : memref<40xf64>
       }{fuse_1}
       %2 = affine.load %arg2[%arg4] {fuse_target1, set=1}: memref<40xf64>
       %3 = affine.load %arg1[%arg4, %arg4] {fuse_target2, set=1}: memref<40x40xf64>
       %4 = arith.divf %2, %3 {fuse_target3, set=1}: f64
       affine.store %4, %arg2[%arg4] {fuse_target4, set=1}: memref<40xf64>
     }{distribute}
     return
   }


  transform.sequence failures(propagate) {
    ^bb1(%arg1: !transform.any_op):
      %1 = transform.structured.match ops{["affine.for"]} attributes {fuse_1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %2 = transform.structured.match attributes {fuse_target1} in %arg1 : (!transform.any_op) -> (!transform.any_op)
      %5 = transform.validator.fuse_into %2 into %1 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)

      %6 = transform.structured.match attributes {fuse_target2} in %arg1 : (!transform.any_op) -> (!transform.any_op)
      %7 = transform.validator.fuse_into %6 into %5 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)

      %8 = transform.structured.match attributes {fuse_target3} in %arg1 : (!transform.any_op) -> (!transform.any_op)
      %9 = transform.validator.fuse_into %8 into %7 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)

      %10 = transform.structured.match attributes {fuse_target4} in %arg1 : (!transform.any_op) -> (!transform.any_op)
      %11 = transform.validator.fuse_into %10 into %9 : (!transform.op<"affine.for">, !transform.any_op) -> (!transform.op<"affine.for">)

      %21 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %211:2 = transform.validator.distribute %21 : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
      %22:4 = transform.validator.tile %211#1 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)

      transform.validator.unroll %22#2 {factor = 8} : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
      transform.validator.parallel %22#1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}

