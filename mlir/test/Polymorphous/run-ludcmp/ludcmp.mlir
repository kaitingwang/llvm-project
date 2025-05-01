//for (int i = 0; i < _PB_N; i++) {
//  for (int j = 0; j < i; j++) {
//    DATA_TYPE w = A[i][j];
//    for (int k = 0; k < j; k++) {
//      w -= A[i][k] * A[k][j];
//    }
//    A[i][j] = w / A[j][j];
//  }
//  #pragma omp parallel for
//  for (int j = i; j < _PB_N; j++) {
//    DATA_TYPE=  w[0] A[i][j];
//    for (int k = 0; k < i; k++) {
//      w -= A[i][k] * A[k][j];
//    }
//    A[i][j] = w;
//  }
//}
//
//for (int i = 0; i < _PB_N; i++) {
//  DATA_TYPE w = b[i];
//  for (int j = 0; j < i; j++)
//    w -= A[i][j] * y[j];
//  y[i] = w;
//}
//
//for (int i = _PB_N - 1; i >= 0; i--) {
//  DATA_TYPE w = y[i];
//  for (int j = i + 1; j < _PB_N; j++)
//    w -= A[i][j] * x[j];
//  x[i] = w / A[i][i];
//}


#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0)>
module attributes {llvm.data_layout = "", llvm.target_triple = ""}  {
  func.func private @ludcmp(%arg0: i32, %arg1: memref<4000x4000xf64>, %arg2: memref<4000xf64>, %arg3: memref<4000xf64>, %arg4: memref<4000xf64>) {
    %0 = arith.index_cast %arg0 : i32 to index
    %w = memref.alloca() {schedule.expand = [1, 2005]}: memref<1xf64>
    %u = memref.alloca() {schedule.expand = [2005]}: memref<1xf64>
    affine.for %arg5 = 0 to %0 {
      affine.for %arg6 = 0 to #map0(%arg5) {
        %2 = affine.load %arg1[%arg5, %arg6] : memref<4000x4000xf64>
        affine.store %2, %w[0] : memref <1xf64>
        affine.for %arg7 = 0 to #map0(%arg6) {
          %6 = affine.load %arg1[%arg5, %arg7] : memref<4000x4000xf64>
          %7 = affine.load %arg1[%arg7, %arg6] : memref<4000x4000xf64>
          %8 = arith.mulf %6, %7 : f64
          %9 = affine.load %w[0] : memref<1xf64>
          %10 = arith.subf %9, %8 : f64
          affine.store %10, %w[0] : memref<1xf64>
        }
        %3 = affine.load %w[0] : memref<1xf64>
        %4 = affine.load %arg1[%arg6, %arg6] : memref<4000x4000xf64>
        %5 = arith.divf %3, %4 : f64
        affine.store %5, %arg1[%arg5, %arg6] : memref<4000x4000xf64>
      }
      affine.for %arg6 = #map0(%arg5) to %0 {
        %2 = affine.load %arg1[%arg5, %arg6] : memref<4000x4000xf64>
        affine.store %2, %w[0] : memref<1xf64>
        affine.for %arg7 = 0 to #map0(%arg5) {
          %4 = affine.load %arg1[%arg5, %arg7] : memref<4000x4000xf64>
          %5 = affine.load %arg1[%arg7, %arg6] : memref<4000x4000xf64>
          %6 = arith.mulf %4, %5 : f64
          %7 = affine.load %w[0] : memref<1xf64>
          %8 = arith.subf %7, %6 : f64
          affine.store %8, %w[0] : memref<1xf64>
        }
        %3 = affine.load %w[0] : memref<1xf64>
        affine.store %3, %arg1[%arg5, %arg6] : memref<4000x4000xf64>
      } {parallel}
    }
    affine.for %arg5 = 0 to %0 {
      %2 = affine.load %arg2[%arg5] : memref<4000xf64>
      affine.store %2, %u[0] : memref<1xf64>
      affine.for %arg6 = 0 to #map0(%arg5) {
        %4 = affine.load %arg1[%arg5, %arg6] : memref<4000x4000xf64>
        %5 = affine.load %arg4[%arg6] : memref<4000xf64>
        %6 = arith.mulf %4, %5 : f64
        %7 = affine.load %u[0] : memref<1xf64>
        %8 = arith.subf %7, %6 : f64
        affine.store %8, %u[0] : memref<1xf64>
      }
      %3 = affine.load %u[0] : memref<1xf64>
      affine.store %3, %arg4[%arg5] : memref<4000xf64>
    }
    affine.for %arg5 = 0 to %0 {
      %2 = affine.load %arg4[-%arg5 + symbol(%0) - 1] : memref<4000xf64>
      affine.store %2, %u[0] : memref<1xf64>
      affine.for %arg6 = #map1(%arg5)[%0] to %0 {
        %6 = affine.load %arg1[-%arg5 + symbol(%0) - 1, %arg6] : memref<4000x4000xf64>
        %7 = affine.load %arg3[%arg6] : memref<4000xf64>
        %8 = arith.mulf %6, %7 : f64
        %9 = affine.load %u[0] : memref<1xf64>
        %10 = arith.subf %9, %8 : f64
        affine.store %10, %u[0] : memref<1xf64>
      }
      %3 = affine.load %u[0] : memref<1xf64>
      %4 = affine.load %arg1[-%arg5 + symbol(%0) - 1, -%arg5 + symbol(%0) - 1] : memref<4000x4000xf64>
      %5 = arith.divf %3, %4 : f64
      affine.store %5, %arg3[-%arg5 + symbol(%0) - 1] : memref<4000xf64>
    }
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %1 = transform.structured.match ops{["affine.for"]} attributes {parallel} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %2 = transform.validator.parallel %1 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}
