add_mlir_dialect_library(MLIRValidatorTransformOps
  ValidatorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Validator/ValidatorTransformOps

  DEPENDS
  MLIRValidatorTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineAnalysis
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRIR
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRTransformDialect
  MLIRVectorDialect
)
