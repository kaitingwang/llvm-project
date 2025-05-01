#map0 = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0] -> (s0 - 1)>
module {
  func.func private @adi(%arg0: i32, %arg1: i32, %arg2: memref<2000x2000xf32>, %arg3: memref<2000x2000xf32>, %arg4: memref<2000x2000xf32>, %arg5: memref<2000x2000xf32>) {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.sitofp %arg1 : i32 to f32
    %2 = arith.divf %cst, %1 : f32
    %3 = arith.sitofp %arg0 : i32 to f32
    %4 = arith.divf %cst, %3 : f32
    %5 = arith.mulf %cst_0, %4 : f32
    %6 = arith.mulf %2, %2 : f32
    %7 = arith.divf %5, %6 : f32
    %8 = arith.mulf %cst, %4 : f32
    %9 = arith.divf %8, %6 : f32
    %10 = arith.negf %7 : f32
    %11 = arith.divf %10, %cst_0 : f32
    %12 = arith.addf %cst, %7 : f32
    %13 = arith.negf %9 : f32
    %14 = arith.divf %13, %cst_0 : f32
    %15 = arith.addf %cst, %9 : f32
    %16 = arith.index_cast %arg0 : i32 to index
    affine.for %arg6 = 1 to #map0()[%16] {
      affine.for %arg7 = 1 to #map1()[%0] {
        affine.store %cst, %arg3[0, %arg7] {set = 0}: memref<2000x2000xf32>
        affine.store %cst_1, %arg4[%arg7, 0] {set = 0}: memref<2000x2000xf32>
        affine.store %cst, %arg5[%arg7, 0] {set = 0}: memref<2000x2000xf32>
        affine.for %arg8 = 1 to #map1()[%0] {
          %17 = arith.negf %11 {set = 1}: f32
          %18 = affine.load %arg4[%arg7, %arg8 - 1] {set = 1}: memref<2000x2000xf32>
          %19 = arith.mulf %11, %18 {set = 1}: f32
          %20 = arith.addf %19, %12 {set = 1}: f32
          %21 = arith.divf %17, %20 {set = 1}: f32
          affine.store %21, %arg4[%arg7, %arg8] {set = 1}: memref<2000x2000xf32>
          %22 = arith.negf %14 {set = 1}: f32
          %23 = affine.load %arg2[%arg8, %arg7 - 1] {set = 1}: memref<2000x2000xf32>
          %24 = arith.mulf %22, %23 {set = 1}: f32
          %25 = arith.mulf %cst_0, %14 {set = 1}: f32
          %26 = arith.addf %cst, %25 {set = 1}: f32
          %27 = affine.load %arg2[%arg8, %arg7] {set = 1}: memref<2000x2000xf32>
          %28 = arith.mulf %26, %27 {set = 1}: f32
          %29 = arith.addf %24, %28 {set = 1}: f32
          %30 = affine.load %arg2[%arg8, %arg7 + 1] {set = 1}: memref<2000x2000xf32>
          %31 = arith.mulf %14, %30 {set = 1}: f32
          %32 = arith.subf %29, %31 {set = 1}: f32
          %33 = affine.load %arg5[%arg7, %arg8 - 1] {set = 1}: memref<2000x2000xf32>
          %34 = arith.mulf %11, %33 {set = 1}: f32
          %35 = arith.subf %32, %34 {set = 1}: f32
          %36 = affine.load %arg4[%arg7, %arg8 - 1] {set = 1}: memref<2000x2000xf32>
          %37 = arith.mulf %11, %36 {set = 1}: f32
          %38 = arith.addf %37, %12 {set = 1}: f32
          %39 = arith.divf %35, %38 {set = 1}: f32
          affine.store %39, %arg5[%arg7, %arg8] {set = 1}: memref<2000x2000xf32>
        }
        affine.store %cst, %arg3[symbol(%0) - 1, %arg7] {set = 2}: memref<2000x2000xf32>
        affine.for %arg8 = 1 to #map1()[%0] {
          %17 = affine.load %arg4[%arg7, -%arg8 + symbol(%0) - 1] {set = 3}: memref<2000x2000xf32>
          %18 = affine.load %arg3[-%arg8 + symbol(%0), %arg7] {set = 3}: memref<2000x2000xf32>
          %19 = arith.mulf %17, %18 {set = 3}: f32
          %20 = affine.load %arg5[%arg7, -%arg8 + symbol(%0) - 1] {set = 3}: memref<2000x2000xf32>
          %21 = arith.addf %19, %20 {set = 3}: f32
          affine.store %21, %arg3[-%arg8 + symbol(%0) - 1, %arg7] {set = 3}: memref<2000x2000xf32>
        }
      } {parallel1, tile1}
      affine.for %arg7 = 1 to #map1()[%0] {
        affine.store %cst, %arg2[%arg7, 0] {set = 0}: memref<2000x2000xf32>
        affine.store %cst_1, %arg4[%arg7, 0] {set = 0}: memref<2000x2000xf32>
        affine.store %cst, %arg5[%arg7, 0] {set = 0}: memref<2000x2000xf32>
        affine.for %arg8 = 1 to #map1()[%0] {
          %17 = arith.negf %14 {set = 1}: f32
          %18 = affine.load %arg4[%arg7, %arg8 - 1] {set = 1}: memref<2000x2000xf32>
          %19 = arith.mulf %14, %18 {set = 1}: f32
          %20 = arith.addf %19, %15 {set = 1}: f32
          %21 = arith.divf %17, %20 {set = 1}: f32
          affine.store %21, %arg4[%arg7, %arg8] {set = 1}: memref<2000x2000xf32>
          %22 = arith.negf %11 {set = 1}: f32
          %23 = affine.load %arg3[%arg7 - 1, %arg8] {set = 1}: memref<2000x2000xf32>
          %24 = arith.mulf %22, %23 {set = 1}: f32
          %25 = arith.mulf %cst_0, %11 {set = 1}: f32
          %26 = arith.addf %cst, %25 {set = 1}: f32
          %27 = affine.load %arg3[%arg7, %arg8] {set = 1}: memref<2000x2000xf32>
          %28 = arith.mulf %26, %27 {set = 1}: f32
          %29 = arith.addf %24, %28 {set = 1}: f32
          %30 = affine.load %arg3[%arg7 + 1, %arg8] {set = 1}: memref<2000x2000xf32>
          %31 = arith.mulf %11, %30 {set = 1}: f32
          %32 = arith.subf %29, %31 {set = 1}: f32
          %33 = affine.load %arg5[%arg7, %arg8 - 1] {set = 1}: memref<2000x2000xf32>
          %34 = arith.mulf %14, %33 {set = 1}: f32
          %35 = arith.subf %32, %34 {set = 1}: f32
          %36 = affine.load %arg4[%arg7, %arg8 - 1] {set = 1}: memref<2000x2000xf32>
          %37 = arith.mulf %14, %36 {set = 1}: f32
          %38 = arith.addf %37, %15 {set = 1}: f32
          %39 = arith.divf %35, %38 {set = 1}: f32
          affine.store %39, %arg5[%arg7, %arg8] {set = 1}: memref<2000x2000xf32>
        }
        affine.store %cst, %arg2[%arg7, symbol(%0) - 1] {set = 2}: memref<2000x2000xf32>
        affine.for %arg8 = 1 to #map1()[%0] {
          %17 = affine.load %arg4[%arg7, -%arg8 + symbol(%0) - 1] {set = 3}: memref<2000x2000xf32>
          %18 = affine.load %arg2[%arg7, -%arg8 + symbol(%0)] {set = 3}: memref<2000x2000xf32>
          %19 = arith.mulf %17, %18 {set = 3}: f32
          %20 = affine.load %arg5[%arg7, -%arg8 + symbol(%0) - 1] {set = 3}: memref<2000x2000xf32>
          %21 = arith.addf %19, %20 {set = 3}: f32
          affine.store %21, %arg2[%arg7, -%arg8 + symbol(%0) - 1] {set = 3}: memref<2000x2000xf32>
        }
      } {parallel2, tile2}
    }
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    // %0 = transform.structured.match ops{["affine.for"]} attributes {parallel1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // %1 = transform.structured.match ops{["affine.for"]} attributes {parallel2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // transform.validator.parallel %0 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    // transform.validator.parallel %1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %0 = transform.structured.match ops{["affine.for"]} attributes {tile1} in %arg1 : (!transform.any_op) -> (!transform.any_op)
    %output:4 = transform.validator.distribute %0 : (!transform.any_op) -> (!transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %1, %2, %3, %4 = transform.validator.tile %output#1 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %5, %6, %7, %8 = transform.validator.tile %output#3 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.reorder %7 and %8 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.parallel %1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %5 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

    %9 = transform.structured.match ops{["affine.for"]} attributes {tile2} in %arg1 : (!transform.any_op) -> (!transform.any_op)
    %10, %11, %12, %13 = transform.validator.distribute %9 : (!transform.any_op) -> (!transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %14, %15, %16, %17 = transform.validator.tile %11 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %18, %19, %20, %21 = transform.validator.tile %13 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    // transform.validator.reorder %18 and %19 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.parallel %14 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.parallel %18 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}