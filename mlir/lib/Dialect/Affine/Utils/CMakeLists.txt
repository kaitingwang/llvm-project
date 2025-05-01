SET(CLOOG_INT_FLAG    "-DCLOOG_INT_GMP")
add_definitions(${CLOOG_INT_FLAG})

link_libraries("/home/amy/llvm-project/cloog-isl/.libs/libcloog-isl.so.4.0.0")
include_directories("/home/amy/llvm-project/cloog-isl/include")

add_mlir_dialect_library(MLIRAffineUtils
  LoopFusionUtils.cpp
  LoopUtils.cpp
  Utils.cpp
  ViewLikeInterfaceUtils.cpp
  PolyToolUtils.cpp
  PolyToolCorrectionUtils.cpp
  PolyToolPrinterUtils.cpp


  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRArithUtils
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTransformUtils
  MLIRViewLikeInterface
  )
