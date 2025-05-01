// RUN: mlir-opt --correct-schedule --split-input-file %s 

#map = affine_map<(d0) -> (d0)>
module {
//   func.func @durbin(%_PB_N: index, %r: memref<?xf32>, %y: memref<?xf32>,
//                         %z: memref<?xf32>, %alpha_mem: memref<1xf32>, %beta_mem: memref<1xf32>, %sum: memref<1xf32>) {

//       %r0 = affine.load %r[0] : memref<?xf32>
//       %a = arith.negf %r0 : f32
//       affine.store %a, %y[0] : memref<?xf32>

//       affine.store %a, %alpha_mem[0] : memref<1xf32>

//       %one1 = arith.constant 1.0 : f32
//       affine.store %one1, %beta_mem[0] : memref<1xf32>

//       affine.for %k = 1 to %_PB_N  {
//           // beta = (1-alpha*alpha)*beta; 
//           %beta = affine.load %beta_mem[0] : memref<1xf32>
//           %alpha = affine.load %alpha_mem[0] : memref<1xf32>
//           %alpha2 = arith.mulf %alpha, %alpha : f32
//           %one = arith.constant 1.0 : f32
//           %1 = arith.subf %one, %alpha2 : f32
//           %2 = arith.mulf %1, %beta : f32 // beta
//           affine.store %2, %beta_mem[0] : memref<1xf32>
//           //sum = 0
//           %zero = arith.constant 0.0 : f32
//           affine.store %zero, %sum[0] : memref<1xf32>

//           affine.for %i = 0 to #map1(%k) {
//               %3 = affine.load %r[%k - %i - 1] : memref<?xf32>
//               %4 = affine.load %y[%i] : memref<?xf32>
//               %5 = arith.mulf %3, %4 : f32
//               %6 = affine.load %sum[0] :  memref<1xf32>
//               %7 = arith.addf %5, %6 : f32
//               affine.store %7, %sum[0] :  memref<1xf32>
//           }

//           // alpha = - (r[k] + sum)/beta;
//           %3 = affine.load %sum[0] :  memref<1xf32>
//           %4 = affine.load %r[%k] : memref<?xf32> //r[k]
//           %b = affine.load %beta_mem[0] : memref<1xf32>
//           %5 = arith.addf %4, %3 : f32 // r[k] + sum
//           %6 = arith.negf %5 : f32 // -(r[k] + sum)
//           %7 = arith.divf %6, %b : f32 // alpha
//           affine.store %7, %alpha_mem[0] : memref<1xf32>
          
//           //z[i] = y[i] + alpha * y[k+i-1]
//           affine.for %i = 0 to #map1(%k) {
//               %33 = affine.load %alpha_mem[0] :  memref<1xf32>
//               %8 = affine.load %y[%i] : memref<?xf32>
//               %9 = affine.load %y[%k - %i - 1] : memref<?xf32>
//               %10 = arith.mulf %33, %9 : f32 
//               %11 = arith.addf %8, %10 : f32
//               affine.store %11, %z[%i] : memref<?xf32>
//           } {fuse_i1}

//           affine.for %i = 0 to #map1(%k) {
//               %12 = affine.load %z[%i] : memref<?xf32>
//               affine.store %12, %y[%i] : memref<?xf32>
//           } {fuse_i2}

//           // y[k] = alpha;
//           %77 = affine.load %alpha_mem[0] :  memref<1xf32>
//           affine.store %77, %y[%k] : memref<?xf32>
//       }

//       return
//   }
  func.func @durbin(%arg0: i32, %arg1: memref<4000xf64>, %arg2: memref<4000xf64>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 0.000000e+00 : f64
    %0 = memref.alloca() : memref<4000xf64>
    %1 = memref.alloca() : memref<1xf64>
    %2 = memref.alloca() : memref<1xf64>
    %3 = memref.alloca() : memref<1xf64>
    %4 = affine.load %arg1[0] : memref<4000xf64>
    %5 = arith.negf %4 : f64
    affine.store %5, %arg2[0] : memref<4000xf64>
    affine.store %cst, %2[0] : memref<1xf64>
    %6 = affine.load %arg1[0] : memref<4000xf64>
    %7 = arith.negf %6 : f64
    affine.store %7, %1[0] : memref<1xf64>
    %8 = arith.index_cast %arg0 : i32 to index
    affine.for %arg3 = 1 to %8 {
      %9 = arith.sitofp %c1_i32 : i32 to f64
      %10 = affine.load %1[0] : memref<1xf64>
      %11 = arith.mulf %10, %10 : f64
      %12 = arith.subf %9, %11 : f64
      %13 = affine.load %2[0] : memref<1xf64>
      %14 = arith.mulf %12, %13 : f64
      affine.store %14, %2[0] : memref<1xf64>
      affine.store %cst_0, %3[0] : memref<1xf64>
      affine.for %arg4 = 0 to #map(%arg3) {
        %20 = affine.load %arg1[%arg3 - %arg4 - 1] : memref<4000xf64>
        %21 = affine.load %arg2[%arg4] : memref<4000xf64>
        %22 = arith.mulf %20, %21 : f64
        %23 = affine.load %3[0] : memref<1xf64>
        %24 = arith.addf %23, %22 : f64
        affine.store %24, %3[0] : memref<1xf64>
      }
      %15 = affine.load %arg1[%arg3] : memref<4000xf64>
      %16 = affine.load %3[0] : memref<1xf64>
      %141 = affine.load %2[0] : memref<1xf64>
      %17 = arith.addf %15, %16 : f64
      %18 = arith.negf %17 : f64
      %19 = arith.divf %18, %141 : f64
      affine.store %19, %1[0] : memref<1xf64>
      affine.for %arg4 = 0 to #map(%arg3) {
        %20 = affine.load %arg2[%arg4] : memref<4000xf64>
        %21 = affine.load %arg2[%arg3 - %arg4 - 1] : memref<4000xf64>
        %191 = affine.load %1[0] : memref<1xf64>
        %22 = arith.mulf %191, %21 : f64
        %23 = arith.addf %20, %22 : f64
        affine.store %23, %0[%arg4] : memref<4000xf64>
      } {fuse_i1}
      affine.for %arg4 = 0 to #map(%arg3) {
        %20 = affine.load %0[%arg4] : memref<4000xf64>
        affine.store %20, %arg2[%arg4] : memref<4000xf64>
      } {fuse_i2}
      %191 = affine.load %1[0] : memref<1xf64>
      affine.store %191, %arg2[%arg3] : memref<4000xf64>
    }
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["affine.for"]} attributes {fuse_i1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %1 = transform.structured.match ops{["affine.for"]} attributes {fuse_i2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
    %2 = transform.validator.fuse %0 with %1 at 1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">)
  }
}

