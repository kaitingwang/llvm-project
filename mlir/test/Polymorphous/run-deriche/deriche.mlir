// static
//void kernel_deriche(int w, int h, DATA_TYPE alpha,
//       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
//       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
//       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
//       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h)) {
//    int i,j;
//    DATA_TYPE xm1, tm1, ym1, ym2;
//    DATA_TYPE xp1, xp2;
//    DATA_TYPE tp1, tp2;
//    DATA_TYPE yp1, yp2;
//
//    DATA_TYPE k;
//    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
//    DATA_TYPE b1, b2, c1, c2;

// #pragma scop
//   k = (SCALAR_VAL(1.0)-math.exp_FUN(-alpha))*(SCALAR_VAL(1.0)-math.exp_FUN(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*math.exp_FUN(-alpha)-math.exp_FUN(SCALAR_VAL(2.0)*alpha));
//   a1 = a5 = k;
//   a2 = a6 = k*math.exp_FUN(-alpha)*(alpha-SCALAR_VAL(1.0));
//   a3 = a7 = k*math.exp_FUN(-alpha)*(alpha+SCALAR_VAL(1.0));
//   a4 = a8 = -k*math.exp_FUN(SCALAR_VAL(-2.0)*alpha);
//   b1 =  POW_FUN(SCALAR_VAL(2.0),-alpha);
//   b2 = -math.exp_FUN(SCALAR_VAL(-2.0)*alpha);
//   c1 = c2 = 1;
//
//   for (i=0; i<_PB_W; i++) {
//        ym1 = SCALAR_VAL(0.0);
//        ym2 = SCALAR_VAL(0.0);
//        xm1 = SCALAR_VAL(0.0);
//        for (j=0; j<_PB_H; j++) {
//           y1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2;
//            xm1 = imgIn[i][j];
//            ym2 = ym1;
//            ym1 = y1[i][j];
//        }
//    }
//
//    for (i=0; i<_PB_W; i++) {
//        yp1 = SCALAR_VAL(0.0);
//        yp2 = SCALAR_VAL(0.0);
//        xp1 = SCALAR_VAL(0.0);
//        xp2 = SCALAR_VAL(0.0);
//        for (j=_PB_H-1; j>=0; j--) {
//            y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
//            xp2 = xp1;
//            xp1 = imgIn[i][j];
//            yp2 = yp1;
//            yp1 = y2[i][j];
//        }
//    }
//
//    for (i=0; i<_PB_W; i++)
//        for (j=0; j<_PB_H; j++) {
//            imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
//        }
//
//    for (j=0; j<_PB_H; j++) {
//        tm1 = SCALAR_VAL(0.0);
//        ym1 = SCALAR_VAL(0.0);
//        ym2 = SCALAR_VAL(0.0);
//        for (i=0; i<_PB_W; i++) {
//            y1[i][j] = a5*imgOut[i][j] + a6*tm1 + b1*ym1 + b2*ym2;
//            tm1 = imgOut[i][j];
//            ym2 = ym1;
//            ym1 = y1 [i][j];
//        }
//    }
//
//
//    for (j=0; j<_PB_H; j++) {
//        tp1 = SCALAR_VAL(0.0);
//        tp2 = SCALAR_VAL(0.0);
//        yp1 = SCALAR_VAL(0.0);
//        yp2 = SCALAR_VAL(0.0);
//        for (i=_PB_W-1; i>=0; i--) {
//            y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
//            tp2 = tp1;
//            tp1 = imgOut[i][j];
//            yp2 = yp1;
//            yp1 = y2[i][j];
//        }
//    }
//
//    for (i=0; i<_PB_W; i++)
//        for (j=0; j<_PB_H; j++)
//            imgOut[i][j] = c2*(y1[i][j] + y2[i][j]);
//
//#pragma endscop
//}





module attributes {}  {
    
  func.func private @deriche(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: memref<64x64xf32>, %arg4: memref<64x64xf32>, %arg5: memref<64x64xf32>, %arg6: memref<64x64xf32>) {
     %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = memref.alloca() { schedule.expand = [4100] }: memref<1xf32>
    %3 = memref.alloca() {schedule.expand = [4100]}: memref<1xf32>
    %4 = memref.alloca() {schedule.expand = [4100]}: memref<1xf32>
    %5 = memref.alloca() {schedule.expand = [4100]}: memref<1xf32>
    %6 = memref.alloca() {schedule.expand = [4100]}: memref<1xf32>
    %7 = memref.alloca() {schedule.expand = [4100]}: memref<1xf32>
    %8 = memref.alloca() {schedule.expand = [4100]}: memref<1xf32>
    %9 = memref.alloca() {schedule.expand = [4100]}: memref<1xf32>
    %10 = memref.alloca() {schedule.expand = [4100]}: memref<1xf32>
    %11 = memref.alloca() {schedule.expand = [4100]}: memref<1xf32>
    %12 = arith.negf %arg2 : f32
    %13 = math.exp %12 : f32
    %14 = arith.subf %cst, %13 : f32
    %15 = arith.mulf %14, %14 : f32
    %16 = arith.mulf %cst_0, %arg2 : f32
    %17 = arith.mulf %16, %13 : f32
    %18 = arith.addf %cst, %17 : f32
    %19 = math.exp %16 : f32
    %20 = arith.subf %18, %19 : f32
    %21 = arith.divf %15, %20 : f32
    %22 = arith.mulf %21, %13 : f32
    %23 = arith.subf %arg2, %cst : f32
    %24 = arith.mulf %22, %23 : f32
    %25 = arith.addf %arg2, %cst : f32
    %26 = arith.mulf %22, %25 : f32
    %27 = arith.negf %21 : f32
    %28 = arith.negf %cst_0 : f32
    %29 = arith.mulf %28, %arg2 : f32
    %30 = math.exp %29 : f32
    %31 = arith.mulf %27, %30 : f32
    %35 = math.powf %cst_0, %12 : f32
    %36 = arith.negf %30 : f32
    %37 = arith.sitofp %c1_i32 : i32 to f32
    affine.for %arg7 = 0 to %1 {
      affine.store %cst_1, %4[0] : memref<1xf32>
      affine.store %cst_1, %5[0] : memref<1xf32>
      affine.store %cst_1, %2[0] : memref<1xf32>
      affine.for %arg8 = 0 to %0 {
        %38 = affine.load %arg3[%arg7, %arg8] : memref<64x64xf32>
        %39 = arith.mulf %21, %38 : f32
        %40 = affine.load %2[0] : memref<1xf32>
        %41 = arith.mulf %24, %40 : f32
        %42 = arith.addf %39, %41 : f32
        %43 = affine.load %4[0] : memref<1xf32>
        %44 = arith.mulf %35, %43 : f32
        %45 = arith.addf %42, %44 : f32
        %46 = affine.load %5[0] : memref<1xf32>
        %47 = arith.mulf %36, %46 : f32
        %48 = arith.addf %45, %47 : f32
        affine.store %48, %arg5[%arg7, %arg8] : memref<64x64xf32>
        affine.store %38, %2[0] : memref<1xf32>
        affine.store %43, %5[0] : memref<1xf32>
        %381 = affine.load %arg5[%arg7, %arg8] : memref<64x64xf32>
        affine.store %381, %4[0] : memref<1xf32>
      }
    } {fuse_1}
    affine.for %arg7 = 0 to %1 {
      affine.store %cst_1, %10[0] : memref<1xf32>
      affine.store %cst_1, %11[0] : memref<1xf32>
      affine.store %cst_1, %6[0] : memref<1xf32>
      affine.store %cst_1, %7[0] : memref<1xf32>
      affine.for %arg8 = 0 to %0 {
        %38 = affine.load %6[0] : memref<1xf32>
        %39 = arith.mulf %26, %38 : f32
        %40 = affine.load %7[0] : memref<1xf32>
        %41 = arith.mulf %31, %40 : f32
        %42 = arith.addf %39, %41 : f32
        %43 = affine.load %10[0] : memref<1xf32>
        %44 = arith.mulf %35, %43 : f32
        %45 = arith.addf %42, %44 : f32
        %46 = affine.load %11[0] : memref<1xf32>
        %47 = arith.mulf %36, %46 : f32
        %48 = arith.addf %45, %47 : f32
        affine.store %48, %arg6[%arg7, -%arg8 + symbol(%0) - 1] : memref<64x64xf32>
        affine.store %38, %7[0] : memref<1xf32>
        %49 = affine.load %arg3[%arg7, -%arg8 + symbol(%0) - 1] : memref<64x64xf32>
        affine.store %49, %6[0] : memref<1xf32>
        affine.store %43, %11[0] : memref<1xf32>
        %381 = affine.load %arg6[%arg7, -%arg8 + symbol(%0) - 1] : memref<64x64xf32>
        affine.store %381, %10[0] : memref<1xf32>
      }
    } {fuse_2}
    affine.for %arg7 = 0 to %1 {
      affine.for %arg8 = 0 to %0 {
        %38 = affine.load %arg5[%arg7, %arg8] : memref<64x64xf32>
        %39 = affine.load %arg6[%arg7, %arg8] : memref<64x64xf32>
        %40 = arith.addf %38, %39 : f32
        %41 = arith.mulf %37, %40 : f32
        affine.store %41, %arg4[%arg7, %arg8] : memref<64x64xf32>
      }
    } {fuse_3}
    affine.for %arg7 = 0 to %0 {
      affine.store %cst_1, %3[0] {set = 0}: memref<1xf32>
      affine.store %cst_1, %4[0] {set = 0}: memref<1xf32>
      affine.store %cst_1, %5[0] {set = 0}: memref<1xf32>
      affine.for %arg8 = 0 to %1 {
        %38 = affine.load %arg4[%arg8, %arg7] {set = 1}: memref<64x64xf32>
        %39 = arith.mulf %21, %38 {set = 1}: f32
        %40 = affine.load %3[0] {set = 1}: memref<1xf32>
        %41 = arith.mulf %24, %40 {set = 1}: f32
        %42 = arith.addf %39, %41 {set = 1}: f32
        %43 = affine.load %4[0] {set = 1}: memref<1xf32>
        %44 = arith.mulf %35, %43 {set = 1}: f32
        %45 = arith.addf %42, %44 {set = 1}: f32
        %46 = affine.load %5[0] {set = 1}: memref<1xf32>
        %47 = arith.mulf %36, %46 {set = 1}: f32
        %48 = arith.addf %45, %47 {set = 1}: f32
        affine.store %48, %arg5[%arg8, %arg7] {set = 1}: memref<64x64xf32>
        affine.store %38, %3[0] {set = 1}: memref<1xf32>
        affine.store %43, %5[0] {set = 1}: memref<1xf32>
        %49 = affine.load %arg5[%arg8, %arg7] {set = 1}: memref<64x64xf32>
        affine.store %49, %4[0] {set = 1}: memref<1xf32>
      }
    } {distribute1}
    affine.for %arg7 = 0 to %0 {
      affine.store %cst_1, %8[0] {set = 0}: memref<1xf32>
      affine.store %cst_1, %9[0] {set = 0}: memref<1xf32>
      affine.store %cst_1, %10[0] {set = 0}: memref<1xf32>
      affine.store %cst_1, %11[0] {set = 0}: memref<1xf32>
      affine.for %arg8 = 0 to %1 {
        %38 = affine.load %8[0] {set = 1}: memref<1xf32>
        %39 = arith.mulf %26, %38 {set = 1}: f32
        %40 = affine.load %9[0] {set = 1}: memref<1xf32>
        %41 = arith.mulf %31, %40 {set = 1}: f32
        %42 = arith.addf %39, %41 {set = 1}: f32
        %43 = affine.load %10[0] {set = 1}: memref<1xf32>
        %44 = arith.mulf %35, %43 {set = 1}: f32
        %45 = arith.addf %42, %44 {set = 1}: f32
        %46 = affine.load %11[0] {set = 1}: memref<1xf32>
        %47 = arith.mulf %36, %46 {set = 1}: f32
        %48 = arith.addf %45, %47 {set = 1}: f32
        affine.store %48, %arg6[-%arg8 + symbol(%1) - 1, %arg7] {set = 1}: memref<64x64xf32>
        affine.store %38, %9[0] {set = 1}: memref<1xf32>
        %49 = affine.load %arg4[-%arg8 + symbol(%1) - 1, %arg7] {set = 1}: memref<64x64xf32>
        affine.store %49, %8[0] {set = 1}: memref<1xf32>
        affine.store %43, %11[0] {set = 1}: memref<1xf32>
        %50 = affine.load %arg6[-%arg8 + symbol(%1) - 1, %arg7] {set = 1}: memref<64x64xf32>
        affine.store %50, %10[0] {set = 1}: memref<1xf32>
      }
    } {distribute2}
    affine.for %arg7 = 0 to %1 {
      affine.for %arg8 = 0 to %0 {
        %38 = affine.load %arg5[%arg7, %arg8] : memref<64x64xf32>
        %39 = affine.load %arg6[%arg7, %arg8] : memref<64x64xf32>
        %40 = arith.addf %38, %39 : f32
        %41 = arith.mulf %37, %40 : f32
        affine.store %41, %arg4[%arg7, %arg8] : memref<64x64xf32>
      }
    } {fuse_6}
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
      %3 = transform.structured.match ops{["affine.for"]} attributes {fuse_1} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %4 = transform.structured.match ops{["affine.for"]} attributes {fuse_2} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %5 = transform.validator.fuse %4 with %3 at 1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">)
      %6 = transform.structured.match ops{["affine.for"]} attributes {fuse_3} in %arg1 : (!transform.any_op) -> !transform.op<"affine.for">
      %7 = transform.validator.fuse %5 with %6 at 1 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">)
      %8 = transform.validator.parallel %7 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
      
      %20 = transform.structured.match ops{["affine.for"]} attributes {distribute1} in %arg1 : (!transform.any_op) -> !transform.any_op
      %21, %22 = transform.validator.distribute %20 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
      %23, %24, %25, %26 = transform.validator.tile %22 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
      %27, %28 = transform.validator.reorder %25 and %26 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
      %29 = transform.validator.parallel %23 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
      
      %30 = transform.structured.match ops{["affine.for"]} attributes {distribute2} in %arg1 : (!transform.any_op) -> !transform.any_op
      %31, %32 = transform.validator.distribute %30 : (!transform.any_op) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
      %33, %34, %35, %36 = transform.validator.tile %32 { tile_sizes = [32, 32] } : (!transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.any_op, !transform.op<"affine.for">, !transform.op<"affine.for">)
      %37, %38 = transform.validator.reorder %35 and %36 : (!transform.op<"affine.for">, !transform.op<"affine.for">) -> (!transform.op<"affine.for">, !transform.op<"affine.for">)
      %39 = transform.validator.parallel %33 : (!transform.op<"affine.for">) -> !transform.op<"affine.for">
      
  }
}
