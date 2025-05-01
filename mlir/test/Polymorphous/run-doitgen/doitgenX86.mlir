module {
  func.func @doitgen(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<250x220x270xf64>, %arg4: memref<270x270xf64>, %arg5: memref<270xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %2 {
      affine.for %arg7 = 0 to %0 {
        affine.for %arg8 = 0 to %1 {
          affine.store %cst, %arg5[%arg8] {set = 0}: memref<270xf64>
          affine.for %arg9 = 0 to %1 {
            %3 = affine.load %arg3[%arg6, %arg7, %arg9] {set = 1, reorder}: memref<250x220x270xf64>
            %4 = affine.load %arg4[%arg9, %arg8] {set = 1, reorder}: memref<270x270xf64>
            %5 = arith.mulf %3, %4 {set = 1, reorder}: f64
            %6 = affine.load %arg5[%arg8] {set = 1, reorder}: memref<270xf64>
            %7 = arith.addf %6, %5 {set = 1, reorder}: f64
            affine.store %7, %arg5[%arg8] {set = 1, reorder}: memref<270xf64>
          } {reorderLoop1}
        } {distribute, reorderLoop2}
        affine.for %arg8 = 0 to %1 {
          %3 = affine.load %arg5[%arg8] : memref<270xf64>
          affine.store %3, %arg3[%arg6, %arg7, %arg8] : memref<250x220x270xf64>
        }
      }
    } 
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %1 = transform.structured.match ops{["affine.for"]} attributes {reorderLoop1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %2 = transform.structured.match ops{["affine.for"]} attributes {reorderLoop2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    transform.validator.partial_reorder %1 and %2  : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> ()
    
    %3 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4, %5 = transform.validator.distribute %3 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)

    //%6 = transform.structured.match ops{["affine.for"]} attributes {reorderLoop1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    transform.validator.unroll %5 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}

