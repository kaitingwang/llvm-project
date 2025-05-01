//#pragma scop
//  for (i = 0; i < _PB_N; i++)
//    y[i] = 0;
//  for (i = 0; i < _PB_M; i++) {
//    tmp[i] = SCALAR_VAL(0.0);
//    for (j = 0; j < _PB_N; j++)
//      tmp[i] = tmp[i] + A[i][j] * x[j];
//    for (j = 0; j < _PB_N; j++)
//      y[j] = y[j] + A[i][j] * tmp[i];
//  }
//#pragma endscop

module attributes {llvm.data_layout = "", llvm.target_triple = ""}  {
   func.func private @atax(%arg0: i32, %arg1: i32, %arg2: memref<?x42xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) {
     %c0_i32 = arith.constant 0 : i32
     %cst = arith.constant 0.000000e+00 : f64
     %0 = arith.index_cast %arg1 : i32 to index
     affine.for %arg6 = 0 to %0 {
       %2 = arith.sitofp %c0_i32 : i32 to f64
       affine.store %2, %arg4[%arg6] : memref<?xf64>
     }
     %1 = arith.index_cast %arg0 : i32 to index
     affine.for %arg6 = 0 to %1 {
       affine.store %cst, %arg5[%arg6] {set = 0} : memref<?xf64>
       affine.for %arg7 = 0 to %0 {
         %2 = affine.load %arg5[%arg6] : memref<?xf64>
         %3 = affine.load %arg2[%arg6, %arg7] : memref<?x42xf64>
         %4 = affine.load %arg3[%arg7] : memref<?xf64>
         %5 = arith.mulf %3, %4 : f64
         %6 = arith.addf %2, %5 : f64
         affine.store %6, %arg5[%arg6] : memref<?xf64>
       } {set = 1}
       affine.for %arg7 = 0 to %0 {
         %2 = affine.load %arg4[%arg7] : memref<?xf64>
         %3 = affine.load %arg2[%arg6, %arg7] : memref<?x42xf64>
         %4 = affine.load %arg5[%arg6] : memref<?xf64>
         %5 = arith.mulf %3, %4 : f64
         %6 = arith.addf %2, %5 : f64
         affine.store %6, %arg4[%arg7] : memref<?xf64>
       } {set = 2}
     } {distribute}
     return
   }
   transform.sequence failures(propagate) {
   ^bb1(%arg1: !transform.any_op):
     %1 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> (!transform.any_op)
     %output:3 = transform.validator.distribute %1 : (!transform.any_op) -> (!transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
     %tiled:4 = transform.validator.tile %output#1 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
     %tiled2:4 = transform.validator.tile %output#2 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
     //%reorder:2 = transform.validator.reorder %tiled#0 and %tiled#1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
     %reorder2:2 = transform.validator.reorder %tiled2#0 and %tiled2#1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
     %parallel = transform.validator.parallel %tiled#0 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
     %parallel2 = transform.validator.parallel %reorder2#0 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

     transform.validator.unroll %tiled#2 { factor = 2 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
     transform.validator.unroll %tiled2#2 { factor = 2 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
   }
}
