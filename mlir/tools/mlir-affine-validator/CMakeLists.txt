list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
 
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
 
# Build tool to compile mlir
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
 
set(DEP_LIBS
  ${conversion_libs}
  ${dialect_libs}
  ${extension_libs}
  ${translation_libs}
  MLIRExecutionEngine
)
 
set(LIBS
  ${DEP_LIBS}
  stdc++fs
)
add_llvm_tool(mlir-affine-validator
  mlir-affine-validator.cpp
 
  DEPENDS
  ${DEP_LIBS}
)
llvm_update_compile_flags(mlir-affine-validator)
target_link_libraries(mlir-affine-validator PRIVATE ${LIBS})
mlir_check_all_link_libraries(mlir-affine-validator)
