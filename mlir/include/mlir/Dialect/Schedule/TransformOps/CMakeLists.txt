set(LLVM_TARGET_DEFINITIONS ValidatorTransformOps.td)
mlir_tablegen(ValidatorTransformOps.h.inc -gen-op-decls)
mlir_tablegen(ValidatorTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRValidatorTransformOpsIncGen)

add_mlir_doc(AffineTransformOps ValidatorLoopTransformOps Dialects/ -gen-op-doc)