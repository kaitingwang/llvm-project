SET(CLOOG_INT_FLAG    "-DCLOOG_INT_GMP")
add_definitions(${CLOOG_INT_FLAG})

link_libraries("/home/amy/llvm-project/cloog-isl/.libs/libcloog-isl.so.4.0.0")
include_directories("/home/amy/llvm-project/cloog-isl/include")

add_mlir_dialect_library(MLIRAffineTransforms
  AffineDataCopyGeneration.cpp
  AffineExpandIndexOps.cpp
  AffineExpandIndexOpsAsAffine.cpp
  AffineLoopInvariantCodeMotion.cpp
  AffineLoopNormalize.cpp
  AffineParallelize.cpp
  AffineScalarReplacement.cpp
  DecomposeAffineOps.cpp
  LoopCoalescing.cpp
  LoopFusion.cpp
  LoopTiling.cpp
  LoopUnroll.cpp
  LoopUnrollAndJam.cpp
  PipelineDataTransfer.cpp
  ReifyValueBounds.cpp
  SuperVectorize.cpp
  SimplifyAffineStructures.cpp
  PolyTool.cpp
  StatementInfo.cpp
  TG.cpp
  ValidatorArrayExpand.cpp
  

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineOpsIncGen
  MLIRAffinePassIncGen
  MLIRLoopLikeInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRAffineUtils
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFUtils
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRValueBoundsOpInterface
  MLIRVectorDialect
  MLIRVectorUtils
  )

