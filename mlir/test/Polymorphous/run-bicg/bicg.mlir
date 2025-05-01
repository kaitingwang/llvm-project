#map1 = affine_map<(d0) -> (d0 + 1)>
module attributes {}  {
  func.func private @bicg(%arg0: i32, %arg1: i32, %arg2: memref<2200x1800xf32>, %arg3: memref<1800xf32>, %arg4: memref<2200xf32>, %arg5: memref<1800xf32>, %arg6: memref<2200xf32>) {
    %c0_f32 = arith.constant 0.0 : f32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %0 {
      affine.store %c0_f32, %arg3[%arg7] : memref<1800xf32>
    }
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg7 = 0 to %1 {
      affine.store %cst, %arg4[%arg7] {set = 0} : memref<2200xf32>
      affine.for %arg8 = 0 to %0 {
        %2 = affine.load %arg3[%arg8] {set = 1} : memref<1800xf32>
        %3 = affine.load %arg6[%arg7] {set = 1} : memref<2200xf32>
        %4 = affine.load %arg2[%arg7, %arg8] {set = 1} : memref<2200x1800xf32>
        %5 = arith.mulf %3, %4 {set = 1} : f32
        %6 = arith.addf %2, %5 {set = 1} : f32
        affine.store %6, %arg3[%arg8] {set = 1} : memref<1800xf32>
        %42 = affine.load %arg2[%arg7, %arg8] {set = 2} : memref<2200x1800xf32>
        %7 = affine.load %arg4[%arg7] {set = 2} : memref<2200xf32>
        %8 = affine.load %arg5[%arg8] {set = 2} : memref<1800xf32>
        %9 = arith.mulf %42, %8 {set = 2} : f32
        %10 = arith.addf %7, %9 {set = 2} : f32
        affine.store %10, %arg4[%arg7] {set = 2} : memref<2200xf32>
      }
    } {distribute}
 
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %2 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> !transform.any_op
    %output:3 = transform.validator.distribute %2 : (!transform.any_op) -> (!transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
 
    %tiled1:4 = transform.validator.tile %output#1 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %tiled2:4 = transform.validator.tile %output#2 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.any_op)
    
    %reorder:2 = transform.validator.reorder %tiled1#0 and %tiled1#1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.parallel %reorder#1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %tiled2#0 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %tiled1#2 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.vectorize %tiled1#3 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %tiled2#2 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}