#map = affine_map<()[s0] -> (s0 - 1)>
module {
  func.func private @fdtd(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<2000x2600xf64>, %arg4: memref<2000x2600xf64>, %arg5: memref<2000x2600xf64>, %arg6: memref<1000xf64>) {
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.69999999999999996 : f64
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg7 = 0 to %2 {
      affine.for %arg8 = 0 to %0 {
        %3 = affine.load %arg6[%arg7] : memref<1000xf64>
        affine.store %3, %arg4[0, %arg8] : memref<2000x2600xf64>
      } {parallel1}
      affine.for %arg8 = 1 to %1 {
        affine.for %arg9 = 0 to %0 {
          %3 = affine.load %arg4[%arg8, %arg9] : memref<2000x2600xf64>
          %4 = affine.load %arg5[%arg8, %arg9] : memref<2000x2600xf64>
          %5 = affine.load %arg5[%arg8 - 1, %arg9] : memref<2000x2600xf64>
          %6 = arith.subf %4, %5 : f64
          %7 = arith.mulf %cst, %6 : f64
          %8 = arith.subf %3, %7 : f64
          affine.store %8, %arg4[%arg8, %arg9] : memref<2000x2600xf64>
        }
      } {parallel2}
      affine.for %arg8 = 0 to %1 {
        affine.for %arg9 = 1 to %0 {
          %3 = affine.load %arg3[%arg8, %arg9] : memref<2000x2600xf64>
          %4 = affine.load %arg5[%arg8, %arg9] : memref<2000x2600xf64>
          %5 = affine.load %arg5[%arg8, %arg9 - 1] : memref<2000x2600xf64>
          %6 = arith.subf %4, %5 : f64
          %7 = arith.mulf %cst, %6 : f64
          %8 = arith.subf %3, %7 : f64
          affine.store %8, %arg3[%arg8, %arg9] : memref<2000x2600xf64>
        }
      } {parallel3}
      affine.for %arg8 = 0 to #map()[%1] {
        affine.for %arg9 = 0 to #map()[%0] {
          %3 = affine.load %arg5[%arg8, %arg9] : memref<2000x2600xf64>
          %4 = affine.load %arg3[%arg8, %arg9 + 1] : memref<2000x2600xf64>
          %5 = affine.load %arg3[%arg8, %arg9] : memref<2000x2600xf64>
          %6 = arith.subf %4, %5 : f64
          %7 = affine.load %arg4[%arg8 + 1, %arg9] : memref<2000x2600xf64>
          %8 = arith.addf %6, %7 : f64
          %9 = affine.load %arg4[%arg8, %arg9] : memref<2000x2600xf64>
          %10 = arith.subf %8, %9 : f64
          %11 = arith.mulf %cst_0, %10 : f64
          %12 = arith.subf %3, %11 : f64
          affine.store %12, %arg5[%arg8, %arg9] : memref<2000x2600xf64>
        }
      } {parallel4}
    } {tile}
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    // %2 = transform.structured.match ops{["affine.for"]} attributes {parallel1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // %3 = transform.validator.parallel %2 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    
    // %4 = transform.structured.match ops{["affine.for"]} attributes {parallel2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // %15, %16, %17, %18 = transform.validator.tile %4 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    // %5 = transform.validator.parallel %15 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    
    // %6 = transform.structured.match ops{["affine.for"]} attributes {parallel3} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // // %151, %161, %171, %181 = transform.validator.tile %6 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    // %7 = transform.validator.parallel %6 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

    // %8 = transform.structured.match ops{["affine.for"]} attributes {parallel4} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    // %152, %162, %172, %182 = transform.validator.tile %8 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    // %9 = transform.validator.parallel %152 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    // %15, %16, %17, %18, %19, %20 = transform.validator.tile %2 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)

    %4 = transform.structured.match ops{["affine.for"]} attributes {parallel2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %6 = transform.structured.match ops{["affine.for"]} attributes {parallel3} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %8 = transform.structured.match ops{["affine.for"]} attributes {parallel4} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %fused = transform.validator.fuse %8 with %6 at 1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> !transform.op<"affine.for">
    %fused2 = transform.validator.fuse %fused with %4 at 1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> !transform.op<"affine.for">
    %152, %162, %172, %182 = transform.validator.tile %fused2 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %9 = transform.validator.parallel %152: (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}

