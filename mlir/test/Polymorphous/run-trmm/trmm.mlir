#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func private @trmm(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<2000x2000xf64>, %arg4: memref<2000x2600xf64>) {
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg5 = 0 to %0 {
      affine.for %arg6 = 0 to %1 {
        affine.for %arg7 = #map(%arg5) to %0 {
          %4 = affine.load %arg3[%arg7, %arg5] {set = 0}: memref<2000x2000xf64>
          %5 = affine.load %arg4[%arg7, %arg6] {set = 0}: memref<2000x2600xf64>
          %6 = arith.mulf %4, %5 {set = 0}: f64
          %7 = affine.load %arg4[%arg5, %arg6] {set = 0}: memref<2000x2600xf64>
          %8 = arith.addf %7, %6 {set = 0}: f64
          affine.store %8, %arg4[%arg5, %arg6] {set = 0}: memref<2000x2600xf64>
        }
        %2 = affine.load %arg4[%arg5, %arg6] {set = 1}: memref<2000x2600xf64>
        %3 = arith.mulf %arg2, %2 {set = 1}: f64
        affine.store %3, %arg4[%arg5, %arg6] {set = 1}: memref<2000x2600xf64>
      }
    } {tile}
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %13 = transform.structured.match ops{["affine.for"]} attributes {tile} in %arg1 : (!transform.any_op) -> !transform.any_op
    %14, %15 = transform.validator.distribute %13 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %16, %17, %18, %19, %20, %21 = transform.validator.tile %14 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %22, %23 = transform.validator.reorder %16 and %17 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %25, %26 = transform.validator.reorder %20 and %21 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %24 = transform.validator.parallel %23 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %26 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

    %27, %28, %29, %30 = transform.validator.tile %15 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %31 = transform.validator.parallel %27 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    transform.validator.unroll %29 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}

