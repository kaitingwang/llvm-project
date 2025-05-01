//#pragma scop
//for (i = _PB_N-1; i >= 0; i--) {
//  for (j=i+1; j<_PB_N; j++) {
//
//    if (j-1>=0)
//      table[i][j] = max_score(table[i][j], table[i][j-1]);
//    if (i+1<_PB_N)
//      table[i][j] = max_score(table[i][j], table[i+1][j]);
//
//    if (j-1>=0 && i+1<_PB_N) {
//     /* don't allow adjacent elements to bond */
//      if (i<j-1)
//        table[i][j] = max_score(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
//      else
//        table[i][j] = max_score(table[i][j], table[i+1][j-1]);
//    }
//
//    for (k=i+1; k<j; k++) {
//      table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
//    }
//  }
//}
//#pragma endscop

#map0 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0)>
#map2 = affine_map<(d0) -> (d0)>
#set0 = affine_set<(d0) : (d0 - 1 >= 0)>
#set1 = affine_set<(d0)[s0] : (-d0 + s0 - 2 >= 0)>
#set2 = affine_set<(d0, d1) : (d1 - d0 - 2 >= 0)>
#set3 = affine_set<(d0)[] : (d0 - 1 >= 0)>
#set4 = affine_set<(d0, d1)[s0] : (-1 * (s0 - d1 - d0  + 1) >= 0)>
#set5 = affine_set<(d0, d1)[s0] : ((s0 - d1 - d0) >= 0)>
module {
  func.func @max_score (%arg0: i32, %arg1: i32) -> i32 attributes {skip.poly} {
    %0 = arith.cmpi "sge", %arg0, %arg1 : i32
    %1 = arith.select %0, %arg0, %arg1 : i32
    return %1 : i32
  }
  func.func @match  (%arg0: i8, %arg1: i8) -> i32 attributes {skip.poly} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i8 = arith.constant 3 : i8
    %0 = arith.addi %arg0, %arg1 : i8
    %1 = arith.cmpi "eq", %0, %c3_i8 : i8
    %2 = arith.select %1, %c1_i32, %c0_i32 : i32
    return %2 : i32
  }
  func.func @nussinov(%arg0: i32, %arg1: memref<?xi8>, %arg2: memref<?x?xi32>) {
    %0 =  arith.index_cast %arg0 : i32 to index
    affine.for %arg3 = 0 to %0 {
      //%1 = affine.apply #map0(%arg3)[%0]
      affine.for %arg4 = #map1(%arg3)[%0] to %0 {
        affine.if #set0(%arg4) {
          %2 = affine.load %arg2[symbol(%0) - %arg3 - 1, %arg4] {manual_set1} : memref<?x?xi32>
          %3 = affine.load %arg2[symbol(%0) - %arg3 - 1, %arg4 - 1] {manual_set1} : memref<?x?xi32>
          %4 = func.call @max_score(%2, %3) {manual_set1} : (i32, i32) -> i32
          affine.store %4, %arg2[symbol(%0) - %arg3 - 1, %arg4] {manual_set1} : memref<?x?xi32>
        }
        //affine.if #set1(%1)[%0] {
        affine.if #set3(%arg3)[] {
          %2 = affine.load %arg2[symbol(%0) - %arg3 - 1, %arg4] {manual_set2} : memref<?x?xi32>
          %3 = affine.load %arg2[symbol(%0) - %arg3 - 1 + 1, %arg4] {manual_set2} : memref<?x?xi32>
          %4 = func.call @max_score(%2, %3) {manual_set2} : (i32, i32) -> i32
          affine.store %4, %arg2[symbol(%0) - %arg3 - 1, %arg4] {manual_set2} : memref<?x?xi32>
        }
        affine.if #set0(%arg4) {
          //affine.if #set1(%1)[%0] {
          affine.if #set3(%arg3)[] {
            //affine.if #set2(%1, %arg3) { 
            affine.if #set4(%arg3, %arg4)[%0] { 
              %2 = affine.load %arg1[symbol(%0) - %arg3 - 1] {manual_set3} : memref<?xi8>
              %3 = affine.load %arg1[%arg4] {manual_set3} : memref<?xi8>
              %4 = func.call @match(%2, %3) {manual_set3} : (i8, i8) -> i32
              %5 = affine.load %arg2[symbol(%0) - %arg3 - 1 + 1, %arg4 - 1] {manual_set3} : memref<?x?xi32>
              %6 = arith.addi %5, %4 {manual_set3} : i32
              %7 = affine.load %arg2[symbol(%0) - %arg3 - 1, %arg4] {manual_set3} : memref<?x?xi32>
              %8 = func.call @max_score(%7, %6) {manual_set3} : (i32, i32) -> i32
              affine.store %8, %arg2[symbol(%0) - %arg3 - 1, %arg4] {manual_set3} : memref<?x?xi32>
            } 
            affine.if #set5(%arg3, %arg4)[%0] {
              %2 = affine.load %arg2[symbol(%0) - %arg3 - 1, %arg4] {manual_set4} : memref<?x?xi32>
              %3 = affine.load %arg2[symbol(%0) - %arg3 - 1 + 1, %arg4 - 1] {manual_set4} : memref<?x?xi32>
              %4 = func.call @max_score(%2, %3) {manual_set4} : (i32, i32) -> i32
              affine.store %4, %arg2[symbol(%0) - %arg3 - 1, %arg4] {manual_set4} : memref<?x?xi32>
            }
          }
        }
        affine.for %arg5 = #map1(%arg3)[%0] to #map2(%arg4) {
          %2 = affine.load %arg2[symbol(%0) - %arg3 - 1, %arg5] {manual_set5} : memref<?x?xi32>
          %3 = affine.load %arg2[%arg5 + 1, %arg4] {manual_set5} : memref<?x?xi32>
          %4 = arith.addi %2, %3 {manual_set5} : i32
          %5 = affine.load %arg2[symbol(%0) - %arg3 - 1, %arg4] {manual_set5} : memref<?x?xi32>
          %6 = func.call @max_score(%5, %4) {manual_set5} : (i32, i32) -> i32
          affine.store %6, %arg2[symbol(%0) - %arg3 - 1, %arg4] {manual_set5} : memref<?x?xi32>
        }
      }
    } {tile}
    return
  }
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !transform.any_op):
    %1 = transform.structured.match attributes {manual_set1} in %arg1 : (!transform.any_op) -> (!transform.any_op)
    %2 = transform.structured.match attributes {manual_set2} in %arg1 : (!transform.any_op) -> (!transform.any_op)
    %3 = transform.structured.match attributes {manual_set3} in %arg1 : (!transform.any_op) -> (!transform.any_op)
    %4 = transform.structured.match attributes {manual_set4} in %arg1 : (!transform.any_op) -> (!transform.any_op)
    %5 = transform.structured.match attributes {manual_set5} in %arg1 : (!transform.any_op) -> (!transform.any_op)
    
    transform.validator.manual_set %1 {schedule = [[-1,  1,   0,   0], 
                                                  [0,  1,   0,   0], 
                                                  [0,   0,  0,   0]]} : (!transform.any_op) -> !transform.any_op
    transform.validator.manual_set %2 {schedule = [[-1,  1,   0,   0], 
                                                  [0,  1,   0,   0], 
                                                  [0,   0,  0,   0]]} : (!transform.any_op) -> !transform.any_op
    transform.validator.manual_set %3 {schedule = [[-1,  1,   0,   0], 
                                                  [0,  1,   0,   0], 
                                                  [0,   0,  0,   0]]} : (!transform.any_op) -> !transform.any_op
    transform.validator.manual_set %4 {schedule = [[-1,  1,   0,   0], 
                                                  [0,  1,   0,   0], 
                                                  [0,   0,  0,   0]]} : (!transform.any_op) -> !transform.any_op
    transform.validator.manual_set %5 {schedule = [[-1,  1,   0, 0,   0], 
                                                  [0,  1,   0, 0,   0], 
                                                  [0,   0,  1, 0,   0]]} : (!transform.any_op) -> !transform.any_op
  }
}
