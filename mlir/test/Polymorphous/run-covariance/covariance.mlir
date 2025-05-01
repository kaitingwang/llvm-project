//static
//void kernel_covariance(int m, int n,
//		       DATA_TYPE float_n,
//		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
//		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
//		       DATA_TYPE POLYBENCH_1D(mean,M,m))
//
//  for (j = 0; j < _PB_M; j++)
//  {
//      mean[j] = SCALAR_VAL(0.0);
//      for (i = 0; i < _PB_N; i++)
//        mean[j] += data[i][j];
//      mean[j] /= float_n;
//    }
//
//  for (i = 0; i < _PB_N; i++)
//    for (j = 0; j < _PB_M; j++)
//      data[i][j] -= mean[j];
//
//  for (i = 0; i < _PB_M; i++)
//    for (j = i; j < _PB_M; j++)
//      {
//        cov[i][j] = SCALAR_VAL(0.0);
//        for (k = 0; k < _PB_N; k++)
//	  cov[i][j] += data[k][i] * data[k][j];
//        cov[i][j] /= (float_n - SCALAR_VAL(1.0));
//        cov[j][i] = cov[i][j];

#map0 = affine_map<()[s0] -> (s0)>
#map1 = affine_map<(d0) -> (d0)>
module attributes {}  {
  func.func private @covariance(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<32x28xf32>, %arg4: memref<28x28xf32>, %arg5: memref<32xf32>) {
    %cst = arith.constant 1.000000e-01 : f32
    %c1 = arith.constant 1 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg7 = 0 to %0 {
        affine.store %cst_0, %arg5[%arg7] {set = 0} : memref<32xf32>
        affine.for %arg8 = 0 to %1 {
            %3 = affine.load %arg3[%arg8, %arg7] {set = 1} : memref<32x28xf32>
            %4 = affine.load %arg5[%arg7] {set = 1} : memref<32xf32>
            %5 = arith.addf %3, %4 {set = 1} : f32
            affine.store %5, %arg5[%arg7] {set = 1} : memref<32xf32>
        }
        %6 = affine.load %arg5[%arg7] {set = 2} : memref<32xf32>
        %7 = arith.divf %6, %arg2 {set = 2} : f32
        affine.store %7, %arg5[%arg7] {set = 2} : memref<32xf32>
    } {dist}

    affine.for %arg7 = 0 to %1 {
        affine.for %arg8 = 0 to %0 {
            %8 = affine.load %arg3[%arg7, %arg8] : memref<32x28xf32>
            %9 = affine.load %arg5[%arg8] : memref<32xf32>
            %10 = arith.subf %8, %9 : f32
            affine.store %10, %arg3[%arg7, %arg8] : memref<32x28xf32>
        }
    } {tile_parallel}

    affine.for %arg7 = 0 to %0 {
        affine.for %arg8 = #map1(%arg7) to %0 {
            affine.store %cst_0, %arg4[%arg7, %arg8] {set = 0} : memref<28x28xf32>
            affine.for %arg9 = 0 to %1 {
                %11 = affine.load %arg3[%arg9, %arg7] {set = 1} : memref<32x28xf32>
                %12 = affine.load %arg3[%arg9, %arg8] {set = 1} : memref<32x28xf32>
                %13 = affine.load %arg4[%arg7, %arg8] {set = 1} : memref<28x28xf32>
                %14 = arith.mulf %11, %12 {set = 1} : f32
                %15 = arith.addf %13, %14 {set = 1} : f32
                affine.store %15, %arg4[%arg7, %arg8] {set = 1} : memref<28x28xf32>
            }
            %16 = arith.subf %arg2, %cst_1 {set = 2} : f32
            %17 = affine.load %arg4[%arg7, %arg8] {set = 2}: memref<28x28xf32> 
            %18 = arith.divf %17, %16 {set = 2}: f32
            affine.store %18, %arg4[%arg7, %arg8] {set = 2}: memref<28x28xf32>
            %118 = affine.load %arg4[%arg7, %arg8] {set = 2} : memref<28x28xf32>
            affine.store %118, %arg4[%arg8, %arg7] {set = 2} : memref<28x28xf32>
        }
    } {distribute} 
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %30 = transform.structured.match ops{["affine.for"]} attributes {dist} in %arg1 : (!transform.any_op) -> !transform.any_op
    %31, %32, %33 = transform.validator.distribute %30 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %34, %35, %36, %37 = transform.validator.tile %32 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.reorder %36 and %37 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    %38 = transform.validator.parallel %34 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    
    %39 = transform.structured.match ops{["affine.for"]} attributes {tile_parallel} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %40, %41, %42, %43 = transform.validator.tile %39 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %44 = transform.validator.parallel %40 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">

    %2 = transform.structured.match ops{["affine.for"]} attributes {distribute} in %arg1 : (!transform.any_op) -> !transform.any_op
    %19, %20, %21 = transform.validator.distribute %2 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %22, %23, %24, %25, %26, %27 = transform.validator.tile %20 { tile_sizes = [32, 32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">, !transform.op<"affine.for">)
    %49, %50 = transform.validator.reorder %26 and %27 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
    transform.validator.unroll %50 { factor = 8 } : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %28 = transform.validator.parallel %22 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %29 = transform.validator.parallel %21 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %45 = transform.validator.parallel %19 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %46 = transform.validator.parallel %31 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
    %47 = transform.validator.parallel %33 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
  }
}
