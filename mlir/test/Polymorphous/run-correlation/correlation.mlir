#map0 = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
module attributes {}  {
  func.func private @correlation(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<3000x2600xf32>, %arg4: memref<2600x2600xf32>, %arg5: memref<2600xf32>, %arg6: memref<2600xf32>) {
    %cst = arith.constant 1.000000e-01 : f32
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %1 {
      affine.store %cst_0, %arg5[%arg7] {set = 0} : memref<2600xf32>
      affine.for %arg8 = 0 to %0 {
        %5 = affine.load %arg3[%arg8, %arg7] {set = 1} : memref<3000x2600xf32>
        %6 = affine.load %arg5[%arg7] {set = 1} : memref<2600xf32>
        %7 = arith.addf %6, %5 {set = 1} : f32
        affine.store %7, %arg5[%arg7] {set = 1} : memref<2600xf32>
      }
      %3 = affine.load %arg5[%arg7] {set = 2} : memref<2600xf32>
      %4 = arith.divf %3, %arg2 {set = 2} : f32
      affine.store %4, %arg5[%arg7] {set = 2} : memref<2600xf32>
    } {distribute1}
    affine.for %arg7 = 0 to %1 {
      affine.store %cst_0, %arg6[%arg7] {set = 0} : memref<2600xf32>
      affine.for %arg8 = 0 to %0 {
        %8 = affine.load %arg3[%arg8, %arg7] {set = 1} : memref<3000x2600xf32>
        %9 = affine.load %arg5[%arg7] {set = 1} : memref<2600xf32>
        %10 = arith.subf %8, %9 {set = 1} : f32
        %11 = arith.mulf %10, %10 {set = 1} : f32
        %12 = affine.load %arg6[%arg7] {set = 1} : memref<2600xf32>
        %13 = arith.addf %12, %11 {set = 1} : f32
        affine.store %13, %arg6[%arg7] {set = 1} : memref<2600xf32>
      }
      %3 = affine.load %arg6[%arg7] {set = 2} : memref<2600xf32>
      %4 = arith.divf %3, %arg2 {set = 2} : f32
      affine.store %4, %arg6[%arg7] {set = 2} : memref<2600xf32>
      %41 = affine.load %arg6[%arg7] {set = 2} : memref<2600xf32>
      %5 = math.sqrt %41 {set = 2} : f32
      affine.store %5, %arg6[%arg7] {set = 2} : memref<2600xf32>
      %6 = arith.cmpf "ule", %5, %cst {set = 2} : f32
      %8 = affine.load %arg6[%arg7] {set = 2} : memref<2600xf32>
      %7 = arith.select %6, %cst_1, %8 {set = 2} : f32
      affine.store %7, %arg6[%arg7] {set = 2} : memref<2600xf32>
    } {distribute2}
    affine.for %arg7 = 0 to %0 {
      affine.for %arg8 = 0 to %1 {
        %3 = affine.load %arg5[%arg8] : memref<2600xf32>
        %4 = affine.load %arg3[%arg7, %arg8] : memref<3000x2600xf32>
        %5 = arith.subf %4, %3 : f32
        affine.store %5, %arg3[%arg7, %arg8] : memref<3000x2600xf32>
        %51 = affine.load %arg3[%arg7, %arg8] : memref<3000x2600xf32>
        %6 = math.sqrt %arg2 : f32
        %7 = affine.load %arg6[%arg8] : memref<2600xf32>
        %8 = arith.mulf %6, %7 : f32
        %9 = arith.divf %51, %8 : f32
        affine.store %9, %arg3[%arg7, %arg8] : memref<3000x2600xf32>
      }
    } {parallel}
 
    %2 = arith.subi %1, %c1 : index
    affine.for %arg7 = 0 to #map0()[%1] {
      affine.store %cst_1, %arg4[%arg7, %arg7] {set = 0} : memref<2600x2600xf32>     // 7 0
      affine.for %arg8 = #map1(%arg7) to %1 {
        affine.store %cst_0, %arg4[%arg7, %arg8] {set = 1} : memref<2600x2600xf32>  // 7 1 0   --> 8 0 0
        affine.for %arg9 = 0 to %0 {
          %4 = affine.load %arg3[%arg9, %arg7] {set = 2} : memref<3000x2600xf32>     // 7 1 1 0 ---> 9 0 0 0
          %5 = affine.load %arg3[%arg9, %arg8] {set = 2} : memref<3000x2600xf32>     // 7 1 1 1
          %6 = arith.mulf %4, %5 {set = 2} : f32
          %7 = affine.load %arg4[%arg7, %arg8] {set = 2} : memref<2600x2600xf32>
          %8 = arith.addf %7, %6 {set = 2} : f32
          affine.store %8, %arg4[%arg7, %arg8] {set = 2} : memref<2600x2600xf32>
        }
        %3 = affine.load %arg4[%arg7, %arg8] {set = 3} : memref<2600x2600xf32>     // 7 1 2   ---> 10 0 0
        affine.store %3, %arg4[%arg8, %arg7] {set = 3} : memref<2600x2600xf32>      // 7 1 3   ---> 10 0 1
      }
    } {tile, distribute} 
    affine.store %cst_1, %arg4[symbol(%2), symbol(%2)] : memref<2600x2600xf32>    // 8   --->  11
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %d1 = transform.structured.match ops{["affine.for"]} attributes {distribute1} in %arg1 : (!transform.any_op) -> !transform.any_op
    %d1_res1, %d1_res2, %d1_res3 = transform.validator.distribute %d1 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.parallel %d1_res1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %d1_res3 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    
    %t1_res1, %t1_res2, %t1_res3, %t1_res4 = transform.validator.tile %d1_res2 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %r1_res1, %r1_res2 = transform.validator.reorder %t1_res3 and %t1_res4 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.unroll %r1_res2 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %t1_res1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
 
    %d2 = transform.structured.match ops{["affine.for"]} attributes {distribute2} in %arg1 : (!transform.any_op) -> !transform.any_op
    %d2_res1, %d2_res2, %d2_res3 = transform.validator.distribute %d2 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    
    transform.validator.parallel %d2_res1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %d2_res3 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    
    %t2_res1, %t2_res2, %t2_res3, %t2_res4 = transform.validator.tile %d2_res2 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %r2_res1, %r2_res2 = transform.validator.reorder %t2_res3 and %t2_res4 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.unroll %r2_res2 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %t2_res1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    //%f2 = transform.validator.fuse %d1_res2 with %d2_res2 at 2 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">)

    %2 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> !transform.any_op
    %11, %12, %13, %14 = transform.validator.distribute %2 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.parallel %11 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %12 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %14 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %15, %16, %17, %18, %19, %20 = transform.validator.tile %13 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.any_op, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %30, %31 = transform.validator.reorder %19 and %20 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.unroll %31 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %15 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

    %p1 = transform.structured.match ops{["affine.for"]} attributes {parallel} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    transform.validator.parallel %p1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}