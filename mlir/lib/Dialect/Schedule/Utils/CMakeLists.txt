add_mlir_dialect_library(MLIRScheduleUtils
  PolyToolTransformationUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Schedule/PolyToolTransformOps

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineAnalysis
  MLIRFuncDialect
  MLIRIR
  MLIRAffineUtils
  MLIRMemRefDialect
)
