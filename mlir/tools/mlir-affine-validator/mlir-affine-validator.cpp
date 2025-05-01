//===- ascend-hlc.cpp - MLIR high-level compiler  -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/SourceMgr.h"
#include <experimental/filesystem>
#include <fstream>

#define DEBUG_TYPE "affine-validator"

namespace mlir {
namespace affine_validator {
namespace cl = llvm::cl;
struct Options {
  cl::opt<std::string> inputFile{
      cl::Positional, cl::desc("the input .mlir file"), cl::init("")};

  cl::opt<uint64_t> parallelDims{
      "parallel-dimensions", cl::Optional,
      cl::desc("the dimension that runs in parallel"), cl::init(0)};

  cl::opt<bool> noThreadLocals{
      "no-thread-local", cl::Optional,
      cl::desc("Do not use the thread local in variable definition"),
      cl::init(true)};

  cl::opt<bool> polybenchCodeGen{
      "polybench-codegen", cl::Optional,
      cl::desc("Force certain output C-files to get generated with headers that match polybench"),
      cl::init(false)};

  cl::opt<bool> printAfterAll{"validator-print-after-all",
                              cl::desc("Print IR after each pass"),
                              cl::init(false)};

  cl::opt<std::string> outputFile{"output", cl::Positional,
                                  cl::desc("the output .mlir file"),
                                  cl::init("")};

  cl::opt<std::string> codegenOutputFile{
      "codegen-output", cl::Positional,
      cl::desc("the output .cpp file for poly codegen"), cl::init("")};

  cl::opt<std::string> customCloog{
      "custom-cloog", cl::Positional,
      cl::desc("User specified cloog file to generate code."), cl::init("")};
};
} // namespace affine_validator
} // namespace mlir

using namespace mlir;
using namespace mlir::affine_validator;

int main(int argc, char **argv) {
  // Initializations
  mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  Options options;
  // Handle command line options
  llvm::cl::ParseCommandLineOptions(argc, argv, "mlir-affine-validator\n");
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string errorMessage;
  auto file = openInputFile(options.inputFile, &errorMessage);
  if (!file) {
    llvm::errs() << "Error opening " << options.inputFile << ": "
                 << errorMessage << '\n';
    return EXIT_FAILURE;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  OwningOpRef<ModuleOp> moduleRef =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!moduleRef) {
    llvm::errs() << "Error opening " << options.inputFile << '\n';
    return EXIT_FAILURE;
  }

  ModuleOp module = *moduleRef;
  PassManager modulePM(module.getContext(), ModuleOp::getOperationName(),
                       OpPassManager::Nesting::Implicit);
  std::function<bool(Pass *, Operation *)> shouldPrintAfterPass =
      [](Pass *, Operation *) { return true; };
  std::function<bool(Pass *, Operation *)> shouldPrintBeforePass;
  if (options.printAfterAll)
    context.disableMultithreading();

  LLVM_DEBUG(llvm::dbgs() << "[mlir-affine-validator] Attempt to raise to "
                             "affine and check schedule.\n");
  // Raise to affine through polygeist passes
  // modulePM.addPass(createRaiseSCFToAffinePass());
  // modulePM.addPass(mlir::replaceAffineCFGPass());
  modulePM.addPass(mlir::createCSEPass());

  // expand array first
  modulePM.addPass(mlir::affine::createValidatorArrayExpandPass());
  
  modulePM.addPass(mlir::createCSEPass());
  modulePM.addPass(mlir::createCanonicalizerPass());

  if (options.printAfterAll)
    modulePM.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, true,
                              true, false, llvm::errs());
  PassManager otherPM(module.getContext(), ModuleOp::getOperationName(),
                      OpPassManager::Nesting::Implicit);
  if (options.printAfterAll)
    otherPM.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, true,
                             true, false, llvm::errs());

  // Run refactored polytool pass
  modulePM.addPass(mlir::affine::createPolyToolPass(options.parallelDims,
                                                      options.noThreadLocals, options.polybenchCodeGen, options.codegenOutputFile));
  
  if (failed(modulePM.run(module))) {
    LLVM_DEBUG(llvm::dbgs()
               << "[mlir-affine-validator] Correction algorithm failed.\n");
    // Validator has failed, attempt ConvertIR pass
    otherPM.addPass(mlir::createLowerAffinePass());
    otherPM.addPass(mlir::createSCCPPass());
    otherPM.addPass(mlir::createForToParallelLoopPass());
    LLVM_DEBUG(llvm::dbgs() << "[mlir-affine-validator] Convert IR to parallel "
                               "form before proceeding.\n");
    if (failed(otherPM.run(module))) {
      llvm::errs() << "[mlir-affine-validator] Failed.\n";
      return EXIT_FAILURE;
    }
  }
  LLVM_DEBUG(
      llvm::dbgs() << "[mlir-affine-validator] Code generation succeeded.\n");

  std::string moduleStr;
  llvm::raw_string_ostream ss(moduleStr);
  ss << module;
  std::string name;
  if (!std::string(options.outputFile).empty())
    name = std::experimental::filesystem::path(std::string(options.outputFile));
  else {
    name = std::experimental::filesystem::path(std::string(options.inputFile))
               .stem();
    name += "_transformed.mlir";
  }
  std::ofstream output(name);
  output << moduleStr;
  output.close();

  return EXIT_SUCCESS;
}
