//=== AffineTransformOps.cpp - Implementation of Affine transformation ops ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Schedule/TransformOps/ValidatorTransformOps.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LabelChain.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/PolyToolCorrectionUtils.h"
#include "mlir/Dialect/Affine/PolyToolUtils.h"
#include "mlir/Dialect/Affine/StatementInfo.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/Support/Debug.h"
#include <utility>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::transform;
using namespace mlir::presburger;

#define DEBUG_TYPE "validator-transform-ops"

namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// ValidatorManualSetOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::ValidatorManualSetOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {

  // TODO: Comeback and look at this code, I suspect its rather broken
  LLVM_DEBUG(llvm::dbgs() << "Validator Manual Set op\n");
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);
  Value v1 = getTarget();
  auto payload = state.getPayloadOps(getTarget());

  for (Operation *target : payload) {
    Attribute schedule = getScheduleAttr();
    assert(schedule.isa<ArrayAttr>());
    assert(schedule.cast<ArrayAttr>().size() == 0 ||
           schedule.cast<ArrayAttr>()[0].isa<ArrayAttr>());

    if (!target->getParentOfType<func::FuncOp>()->hasAttr("PolyTool")) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    SmallVector<SmallVector<int64_t>> userSchedule =
        parseMatrix(schedule.cast<ArrayAttr>());
    (*(vstate.StatementInfoMap))[target].userSchedule = userSchedule;

    (*(vstate.StatementInfoMap))[target].manualSetFlag = true;

    results.set(getResult().cast<OpResult>(), {target});
    return DiagnosedSilenceableFailure::success();
  }

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ValidatorUnrollOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::ValidatorUnrollOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {
  LLVM_DEBUG(llvm::dbgs() << "Validator UnrollOp\n");

  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);
  Value v1 = getTarget();
  auto payload = state.getPayloadOps(getTarget());
  AffineForOp loop1 = dyn_cast<AffineForOp>(*payload.begin());
  int loop1Depth = affine::getNestingDepth(loop1);
  Node *TgRoot = (vstate.TgRoot);

  if (!loop1->getParentOfType<func::FuncOp>()->hasAttr("PolyTool")) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  int unroll_factor = getFactor();
  // We need to build nodes for the matchOp in their useCase since we don't
  // have them inside the validator
  if (dyn_cast_or_null<transform::MatchOp>(v1.getDefiningOp())) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loop1Depth, parents, children, loop1, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v1] = tmp;
  }
  Node *nodeLoop1 = (*(vstate.TgMap))[v1];
  loop1Depth = nodeLoop1->getDepth();
  SmallVector<Operation *> affectedStatements =
      nodeLoop1->findAffectedStatements(*(vstate.StatementInfoMap));
  LLVM_DEBUG(llvm::dbgs() << "UNROLL OP ON " << affectedStatements.size()
                          << " STATEMENTS IN PROCESS...\n");
  SmallVector<int> selectedOp;
  for (auto op : affectedStatements) {
    selectedOp.push_back(((*(vstate.StatementInfoMap))[op]).opId);
    ((*(vstate.StatementInfoMap))[op])
        .transformationList.push_back(new LabelUnroll(unroll_factor));
    LLVM_DEBUG(llvm::dbgs() << ((*(vstate.StatementInfoMap))[op]).opId << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "UNROLL OP ON " << affectedStatements.size()
                          << " STATEMENTS ENDS\n");
  v1 = getOperation()->getOpResult(0);

  SmallVector<Node *> parents = {nodeLoop1};
  SmallVector<Node *> children;
  nodeLoop1->isLeaf = false;
  Node *tmp1 = new Node(loop1Depth, parents, children, loop1, true, false,
                        new LabelUnroll(selectedOp, unroll_factor));

  (*(vstate.TgMap))[v1] = tmp1;
  results.set(v1.cast<OpResult>(), {loop1});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ValidatorVectorizeOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::ValidatorVectorizeOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {
  LLVM_DEBUG(llvm::dbgs() << "Validator VectorizeOp\n");
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);
  Value v1 = getTarget();
  auto payload = state.getPayloadOps(getTarget());
  AffineForOp loop1 = dyn_cast<AffineForOp>(*payload.begin());
  int loop1Depth = affine::getNestingDepth(loop1);
  Node *TgRoot = (vstate.TgRoot);

  if (!loop1->getParentOfType<func::FuncOp>()->hasAttr("PolyTool")) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  // We need to build nodes for the matchOp in their useCase since we don't
  // have them inside the validator
  if (dyn_cast_or_null<transform::MatchOp>(v1.getDefiningOp())) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loop1Depth, parents, children, loop1, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v1] = tmp;
  }

  Node *nodeLoop1 = (*(vstate.TgMap))[v1];
  loop1Depth = nodeLoop1->getDepth();
  SmallVector<Operation *> affectedStatements =
      nodeLoop1->findAffectedStatements(*(vstate.StatementInfoMap));
  LLVM_DEBUG(llvm::dbgs() << "VECTORIZE OP ON " << affectedStatements.size()
                          << " STATEMENTS IN PROCESS...\n");
  SmallVector<int> selectedOp;
  for (auto op : affectedStatements) {
    selectedOp.push_back(((*(vstate.StatementInfoMap))[op]).opId);
    ((*(vstate.StatementInfoMap))[op])
        .transformationList.push_back(new LabelVectorize());
    LLVM_DEBUG(llvm::dbgs() << ((*(vstate.StatementInfoMap))[op]).opId << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "VECTORIZE OP ON " << affectedStatements.size()
                          << " STATEMENTS ENDS\n");
  v1 = getOperation()->getOpResult(0);

  SmallVector<Node *> parents = {nodeLoop1};
  SmallVector<Node *> children;
  nodeLoop1->isLeaf = false;
  Node *tmp1 = new Node(loop1Depth, parents, children, loop1, true, false,
                        new LabelVectorize(selectedOp));

  (*(vstate.TgMap))[v1] = tmp1;
  results.set(v1.cast<OpResult>(), {loop1});

  // TODO: do we need schedule modification/correction?
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ValidatorParallelOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::ValidatorParallelOp::apply(transform::TransformRewriter &rewriter,
                                      transform::TransformResults &results,
                                      transform::TransformState &state) {
  LLVM_DEBUG(llvm::dbgs() << "Validator ParallelOp\n");
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);

  Value v1 = getTarget();
  OpBuilder builder(getContext());
  auto payload = state.getPayloadOps(v1);
  AffineForOp loop1 = dyn_cast<AffineForOp>(*payload.begin());
  int loop1Depth = affine::getNestingDepth(loop1);
  Operation *op1 = v1.getDefiningOp();

  if (!loop1->getParentOfType<func::FuncOp>()->hasAttr("PolyTool")) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  Node *TgRoot = (vstate.TgRoot);
  // We need to build nodes for the matchOp in their useCase since we don't
  // have them inside the validator
  if (dyn_cast_or_null<transform::MatchOp>(op1)) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loop1Depth, parents, children, loop1, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v1] = tmp;
  }

  // Nodes corresponding to the Transformation graph for the two loops in the
  // reorder operand
  Node *nodeLoop1 = (*(vstate.TgMap))[v1];
  loop1Depth = nodeLoop1->getDepth();
  for (int i = 0; i < (*(vstate.dependenceEdges)).size(); i++) {
    dependenceEdge *e = &((*(vstate.dependenceEdges))[i]);
    bool srcIsTiled = ((*(vstate.StatementInfoMap))[e->src]).isTiled();
    bool dstIsTiled = ((*(vstate.StatementInfoMap))[e->dst]).isTiled();
    if (opInsideLoop(e->src, loop1) && opInsideLoop(e->dst, loop1) &&
        !dstIsTiled && !srcIsTiled)
      e->parallelDepth = loop1Depth;
    // Remove dependence edges when force parallel is set to true.
    if (getForce() && e->parallelDepth == e->depth)
      e->isEmpty = true;
  }
  SmallVector<Operation *> affectedStatements =
      nodeLoop1->findAffectedStatements(*(vstate.StatementInfoMap));
  LLVM_DEBUG(llvm::dbgs() << "PARALLEL OP ON " << affectedStatements.size()
                          << " STATEMENTS IN PROCESS...\n");
  SmallVector<int> selectedOp;
  for (auto op : affectedStatements) {
    selectedOp.push_back(((*(vstate.StatementInfoMap))[op]).opId);
    ((*(vstate.StatementInfoMap))[op])
        .transformationList.push_back(new LabelParallel());
    if (((*(vstate.StatementInfoMap))[op]).isTiled() && loop1Depth == 1) {
      LLVM_DEBUG(llvm::dbgs() << "APPLY SUM TRICK ON PARALLEL\n");
      ((*(vstate.StatementInfoMap))[op]).applySumTrick(loop1Depth);
    }
    LLVM_DEBUG(llvm::dbgs() << ((*(vstate.StatementInfoMap))[op]).opId << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "PARALLEL OP ON " << affectedStatements.size()
                          << " STATEMENTS ENDS\n");
  v1 = getOperation()->getOpResult(0);

  SmallVector<Node *> parents = {nodeLoop1};
  SmallVector<Node *> children;
  nodeLoop1->isLeaf = false;
  Node *tmp1 = new Node(loop1Depth, parents, children, loop1, true, false,
                        new LabelParallel(selectedOp));
  (*(vstate.TgMap))[v1] = tmp1;

  results.set(cast<OpResult>(getResult()), {loop1});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ValidatorDistributedParallelOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::ValidatorDistributedParallelOp::apply(transform::TransformRewriter &rewriter,
                                      transform::TransformResults &results,
                                      transform::TransformState &state) {
  int nproc = getNproc();
  LLVM_DEBUG(llvm::dbgs() << "Validator DistributedParallelOp: " << nproc << "\n");
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);

  Value v1 = getTarget();
  OpBuilder builder(getContext());
  auto payload = state.getPayloadOps(v1);
  Operation *op = *payload.begin();
  AffineForOp loop1 = dyn_cast<AffineForOp>(op);
  OpBuilder b(getContext());
  op->setAttr("distributed_parallel", b.getI64IntegerAttr(nproc));
  results.set(cast<OpResult>(getResult()), {loop1});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ValidatorDistributeOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::ValidatorDistributeOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  LLVM_DEBUG(llvm::dbgs() << "Validator DistributeOp\n");
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);

  Value v1 = getTarget();
  Operation *op1 = v1.getDefiningOp();
  auto payload = state.getPayloadOps(v1);
  AffineForOp loop = dyn_cast<AffineForOp>(*payload.begin());
  int loopDepth = affine::getNestingDepth(loop);
  Location loc = loop.getLoc();

  Node *TgRoot = (vstate.TgRoot);
  func::FuncOp func = loop->getParentOfType<func::FuncOp>();
  int maxSetNumber = -1;
  if (!func->hasAttr("PolyTool")) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  if (dyn_cast_or_null<transform::MatchOp>(op1)) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loopDepth, parents, children, loop, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v1] = tmp;
  }

  Node *nodeLoop = (*(vstate.TgMap))[v1];
  loopDepth = nodeLoop->getDepth();
  SmallVector<Operation *> affectedStatements =
      nodeLoop->findAffectedStatements(*(vstate.StatementInfoMap));
  if (affectedStatements.size() == 0)
    return DiagnosedSilenceableFailure::success();
  Operation *firstAffectedStatement = affectedStatements[0];
  Operation *lastAffectedStatement = affectedStatements[0];
  for (auto op : affectedStatements) {
    if (hasSmallerConstantRows(
            ((*(vstate.StatementInfoMap))[op]),
            ((*(vstate.StatementInfoMap))[firstAffectedStatement])))
      firstAffectedStatement = op;
    if (hasSmallerConstantRows(
            ((*(vstate.StatementInfoMap))[lastAffectedStatement]),
            ((*(vstate.StatementInfoMap))[op])))
      lastAffectedStatement = op;
  }

  SmallVector<int> LastAffectedBeforeChange =
      ((*(vstate.StatementInfoMap))[lastAffectedStatement]).constantRows;
  SmallVector<int> prev;
  SmallVector<int> now;
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // any statement before the first statement in the distribution has no
    // effect
    if (!hasSmallerConstantRows(
            ((*(vstate.StatementInfoMap))[op]),
            ((*(vstate.StatementInfoMap))[firstAffectedStatement]))) {
      if (hasSmallerConstantRows(LastAffectedBeforeChange,
                                 ((*(vstate.StatementInfoMap))[op]))) {
        // After the distribution set
        bool mustChange = true;
        SmallVector<int> &const1 =
            ((*(vstate.StatementInfoMap))[op]).constantRows;
        for (int i = 0; i < loopDepth; i++)
          if (const1[i] != LastAffectedBeforeChange[i])
            mustChange = false;
        if (mustChange)
          const1[loopDepth] += maxSetNumber;
      } else {
        // Inside the distribution set
        // Since this is going over the function, we can have operations like
        // for loops that does not have set in them
        if (op->hasAttr("set")) {
          int tmp =
              static_cast<int>(op->getAttrOfType<IntegerAttr>("set").getInt());
          SmallVector<int> &const1 =
              ((*(vstate.StatementInfoMap))[op]).constantRows;
          (((*(vstate.StatementInfoMap))[op]).transformationList)
              .push_back(new LabelDistribution(tmp));
          const1[loopDepth] += tmp;
          if (tmp != maxSetNumber) {
            prev = const1;
            for (int i = loopDepth + 1; i < (int)const1.size(); i++)
              const1[i] = 0;
            now = const1;
          } else {
            SmallVector<int> temp = const1;
            for (int i = loopDepth + 1; i < (int)const1.size(); i++) {
              if (const1[i] == prev[i])
                const1[i] = now[i];
              else {
                const1[i] = now[i] + 1;
                break;
              }
            }
            prev = temp;
            now = const1;
          }
          maxSetNumber = std::max(tmp, maxSetNumber);
        }
      }
    }

    // For debug only,
    // It will print the constant rows when we have
    // -debug-only=correct-schedule
    ((*(vstate.StatementInfoMap))[op]).printConstantRow();
  });
  for (int i = 0; i <= maxSetNumber; i++) {
    // create nodes and the return handles
    v1 = getOperation()->getOpResult(i);
    results.set(getOperation()->getOpResult(i).cast<OpResult>(), {loop});
    SmallVector<Node *> parents = {nodeLoop};
    SmallVector<Node *> children;
    nodeLoop->isLeaf = false;

    // building new nodes for the TG
    Node *tmp2 = new Node(loopDepth, parents, children, loop, true, false,
                          new LabelDistribution(i));

    // set the TgMap to the value of the output
    (*(vstate.TgMap))[v1] = tmp2;
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::ValidatorDistributeOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : ";
  p.printFunctionalType(TypeRange(getOperand().getType()),
                        getResults().getTypes());
}

ParseResult transform::ValidatorDistributeOp::parse(OpAsmParser &parser,
                                                    OperationState &result) {
  OpAsmParser::UnresolvedOperand targetOperand;
  if (parser.parseOperand(targetOperand) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  FunctionType trailingType;
  SMLoc typeLoc;
  if (parser.getCurrentLocation(&typeLoc) ||
      parser.parseColonType(trailingType)) {
    return failure();
  }
  if (trailingType.getNumInputs() != 1)
    return parser.emitError(typeLoc) << "expected one input type";

  result.addTypes(trailingType.getResults());
  if (parser.resolveOperand(targetOperand, trailingType.getInput(0),
                            result.operands))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// ValidatorFuseIntoOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::ValidatorFuseIntoOp::apply(transform::TransformRewriter &rewriter,
                                      transform::TransformResults &results,
                                      transform::TransformState &state) {

  LLVM_DEBUG(llvm::dbgs() << "Validator FuseIntoOp\n");
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);
  Value v1 = getLoop();
  Value v2 = getTarget();
  auto payload1 = state.getPayloadOps(v1);
  auto payload2 = state.getPayloadOps(v2);
  AffineForOp loop = dyn_cast<AffineForOp>(*payload1.begin());
  Operation *target = *payload2.begin();
  OpBuilder b(target);
  int loopDepth = affine::getNestingDepth(loop);
  LLVM_DEBUG(llvm::dbgs() << "target:"
                          << loop.getBody()->getOperations().size() - 1
                          << " loopdepth: " << loopDepth << "\n");

  Operation *op1 = v1.getDefiningOp();
  Operation *op2 = v2.getDefiningOp();
  Node *TgRoot = (vstate.TgRoot);
  if (!loop->getParentOfType<func::FuncOp>()->hasAttr("PolyTool")) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  // We need to build nodes for the matchOp in their useCase since we don't
  // have them inside the validator
  if (dyn_cast_or_null<transform::MatchOp>(op1)) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loopDepth, parents, children, loop, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v1] = tmp;
  }

  // Nodes corresponding to the Transformation graph for the two loops in the
  // reorder operand
  Node *nodeLoop1 = (*(vstate.TgMap))[v1];
  loopDepth = nodeLoop1->getDepth();
  SmallVector<Operation *> affectedStatements =
      nodeLoop1->findAffectedStatements(*(vstate.StatementInfoMap));
  LLVM_DEBUG(llvm::dbgs() << "FUSE-INTO OP ON " << affectedStatements.size()
                          << " STATEMENTS IN PROCESS...\n");
  // Fix the constant Rows
  SmallVector<int> lastAffectedStatement =
      ((*(vstate.StatementInfoMap))[affectedStatements.back()]).constantRows;
  lastAffectedStatement[lastAffectedStatement.size() - 1]++;
  ((*(vstate.StatementInfoMap))[target]).constantRows = lastAffectedStatement;

  LLVM_DEBUG(llvm::dbgs() << "isFusedInto increased by "
                          << (loopDepth + 1) - getNestingDepth(target) << "\n");

  LLVM_DEBUG(llvm::dbgs() << "depth is " << loopDepth << "\n");

  ((*(vstate.StatementInfoMap))[target]).isFusedInto +=
      (loopDepth + 1) - getNestingDepth(target);
  ((*(vstate.StatementInfoMap))[target]).printConstantRow();
  ((*(vstate.StatementInfoMap))[target]).printDomain();
  ((*(vstate.StatementInfoMap))[target]).printScattering();

  LLVM_DEBUG(llvm::dbgs() << "FUSE-INTO OP ON " << affectedStatements.size()
                          << " STATEMENTS ENDS\n");
  v1 = getOperation()->getOpResult(0);

  SmallVector<Node *> parents = {nodeLoop1};
  SmallVector<Node *> children;
  nodeLoop1->isLeaf = false;
  Node *tmp1 = new Node(loopDepth, parents, children, loop, true, false,
                        new LabelFuseInto(target));
  (*(vstate.TgMap))[v1] = tmp1;

  results.set(getOperation()->getOpResult(0).cast<OpResult>(), {loop});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ValidatorReorderOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ValidatorReorderOp::apply(transform::TransformRewriter &rewriter,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  LLVM_DEBUG(llvm::dbgs() << "Reorder op starts\n");
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);
  Value v1 = getFirstLoop();
  Value v2 = getSecondLoop();
  OpBuilder builder(getContext());
  IntegerAttr firstLoopLabel, secondLoopLabel;
  std::string attrName("reordered_to");
  auto payload1 = state.getPayloadOps(v1);
  auto payload2 = state.getPayloadOps(v2);
  AffineForOp loop1 = dyn_cast<AffineForOp>(*payload1.begin());
  AffineForOp loop2 = dyn_cast<AffineForOp>(*payload2.begin());
  int loop1Depth = affine::getNestingDepth(loop1),
      loop2Depth = affine::getNestingDepth(loop2);
  Operation *op1 = v1.getDefiningOp();
  Operation *op2 = v2.getDefiningOp();
  Node *TgRoot = (vstate.TgRoot);
  if (!loop1->getParentOfType<func::FuncOp>()->hasAttr("PolyTool")) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  // We need to build nodes for the matchOp in their useCase since we don't
  // have them inside the validator
  if (dyn_cast_or_null<transform::MatchOp>(op1)) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loop1Depth, parents, children, loop1, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v1] = tmp;
  }

  if (dyn_cast_or_null<transform::MatchOp>(op2)) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loop2Depth, parents, children, loop2, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v2] = tmp;
  }

  // Nodes corresponding to the Transformation graph for the two loops in the
  // reorder operand
  Node *nodeLoop1 = (*(vstate.TgMap))[v1];
  Node *nodeLoop2 = (*(vstate.TgMap))[v2];
  loop1Depth = nodeLoop1->getDepth();
  loop2Depth = nodeLoop2->getDepth();
  SmallVector<Operation *> affectedStatements =
      nodeLoop2->findAffectedStatements(*(vstate.StatementInfoMap));
  LLVM_DEBUG(llvm::dbgs() << "REORDER OP ON " << affectedStatements.size()
                          << " STATEMENTS IN PROCESS...\n");
  for (auto op : affectedStatements) {
    ((*(vstate.StatementInfoMap))[op]).LoopReorder(loop1Depth, loop2Depth);
    // ((*(vstate.StatementInfoMap))[op]).printConstantRow();
    // ((*(vstate.StatementInfoMap))[op]).printDomain();
    // ((*(vstate.StatementInfoMap))[op]).printScattering();
  }
  LLVM_DEBUG(llvm::dbgs() << "REORDER OP ON " << affectedStatements.size()
                          << " STATEMENTS ENDS\n");
  v1 = getOperation()->getOpResult(0);
  v2 = getOperation()->getOpResult(1);

  SmallVector<Node *> parents = {nodeLoop1};
  SmallVector<Node *> children;
  nodeLoop1->isLeaf = false;
  Node *tmp1 = new Node(loop2Depth, parents, children, loop1, true, false,
                        new LabelInterchange(loop1Depth, loop2Depth));
  (*(vstate.TgMap))[v1] = tmp1;

  parents = {nodeLoop2};
  nodeLoop2->isLeaf = false;
  Node *tmp2 = new Node(loop1Depth, parents, children, loop2, true, false,
                        new LabelInterchange(loop1Depth, loop2Depth));
  (*(vstate.TgMap))[v2] = tmp2;

  results.set(getOperation()->getOpResult(0).cast<OpResult>(), {loop2});
  results.set(getOperation()->getOpResult(1).cast<OpResult>(), {loop1});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ValidatorPartialReorderOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure transform::ValidatorPartialReorderOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  LLVM_DEBUG(llvm::dbgs() << "Partial Reorder op starts\n");
  // start transform schedule
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);
  Value v1 = getFirstLoop();
  Value v2 = getSecondLoop();
  OpBuilder builder(getContext());
  auto payload1 = state.getPayloadOps(v1);
  auto payload2 = state.getPayloadOps(v2);
  assert(llvm::hasSingleElement(payload1) && llvm::hasSingleElement(payload2) &&
         "expected a single target op");
  AffineForOp loop1 = dyn_cast<AffineForOp>(*payload1.begin());
  AffineForOp loop2 = dyn_cast<AffineForOp>(*payload2.begin());
  int loop1Depth = affine::getNestingDepth(loop1),
      loop2Depth = affine::getNestingDepth(loop2);
  Operation *op1 = v1.getDefiningOp();
  Operation *op2 = v2.getDefiningOp();
  Node *TgRoot = (vstate.TgRoot);
  if (!loop1->getParentOfType<func::FuncOp>()->hasAttr("PolyTool")) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  // We need to build nodes for the matchOp in their useCase since we don't
  // have them inside the validator
  if (dyn_cast_or_null<transform::MatchOp>(op1)) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loop1Depth, parents, children, loop1, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v1] = tmp;
  }

  if (dyn_cast_or_null<transform::MatchOp>(op2)) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loop2Depth, parents, children, loop2, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v2] = tmp;
  }

  // Nodes corresponding to the Transformation graph for the two loops in the
  // reorder operand
  Node *nodeLoop1 = (*(vstate.TgMap))[v1];
  Node *nodeLoop2 = (*(vstate.TgMap))[v2];
  loop1Depth = nodeLoop1->getDepth();
  loop2Depth = nodeLoop2->getDepth();
  SmallVector<Operation *> affectedStatements =
      nodeLoop2->findAffectedStatements(*(vstate.StatementInfoMap));
  LLVM_DEBUG(llvm::dbgs() << "Partial REORDER OP ON "
                          << affectedStatements.size()
                          << " STATEMENTS IN PROCESS...\n");
  for (Operation *op : affectedStatements) {
    if (op->hasAttr("reorder"))
      ((*(vstate.StatementInfoMap))[op]).LoopReorder(loop1Depth, loop2Depth);
  }
  LLVM_DEBUG(llvm::dbgs() << "Partial REORDER OP ON "
                          << affectedStatements.size() << " STATEMENTS ENDS\n");
  // v1 = getOperation()->getOpResult(0);
  // v2 = getOperation()->getOpResult(1);

  SmallVector<Node *> parents = {nodeLoop1};
  SmallVector<Node *> children;
  nodeLoop1->isLeaf = false;
  // Node *tmp1 = new Node(loop2Depth, parents, children, loop1, true, false,
  // new LabelInterchange(loop1Depth, loop2Depth));
  //(*(vstate.TgMap))[v1] = tmp1;

  parents = {nodeLoop2};
  nodeLoop2->isLeaf = false;
  // Node *tmp2 = new Node(loop1Depth, parents, children, loop2, true, false,
  // new LabelInterchange(loop1Depth, loop2Depth));
  //(*(vstate.TgMap))[v2] = tmp2;

  // results.set(getOperation()->getOpResult(0).cast<OpResult>(), {loop2});
  // results.set(getOperation()->getOpResult(1).cast<OpResult>(), {loop1});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ValidatorFuseOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ValidatorFuseOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  LLVM_DEBUG(llvm::dbgs() << "Validator FuseOp\n");
  Value v1 = getTarget1();
  Value v2 = getTarget2();
  auto payload1 = state.getPayloadOps(v1);
  auto payload2 = state.getPayloadOps(v2);
  assert(llvm::hasSingleElement(payload1) && llvm::hasSingleElement(payload2) &&
         "expected a single target op");

  AffineForOp target1 = dyn_cast<AffineForOp>(*payload1.begin());
  AffineForOp target2 = dyn_cast<AffineForOp>(*payload2.begin());
  int loop1Depth = affine::getNestingDepth(target1),
      loop2Depth = affine::getNestingDepth(target2);
  if (target1 == target2)
    return DiagnosedSilenceableFailure::definiteFailure();
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);

  Operation *op1 = v1.getDefiningOp();
  Operation *op2 = v2.getDefiningOp();
  Node *TgRoot = (vstate.TgRoot);
  func::FuncOp func = target1->getParentOfType<func::FuncOp>();
  if (!target1->getParentOfType<func::FuncOp>()->hasAttr("PolyTool")) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  if (dyn_cast_or_null<transform::MatchOp>(op1)) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loop1Depth, parents, children, target1, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v1] = tmp;
  }

  if (dyn_cast_or_null<transform::MatchOp>(op2)) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loop2Depth, parents, children, target2, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v2] = tmp;
  }

  // Nodes corresponding to the Transformation graph for the two loops in the
  // reorder operand
  Node *nodeLoop1 = (*(vstate.TgMap))[v1];
  Node *nodeLoop2 = (*(vstate.TgMap))[v2];

  // fusion depth
  // this number should be less that the depth of all the statements
  // otherwise it's an assertion
  int fusionDepth = getDepth();
  fusionDepth--;

  // anyway for fusion these two loop depths must be identical
  loop1Depth = nodeLoop1->getDepth();
  loop2Depth = nodeLoop2->getDepth();

  // all the statements inside the first loops
  SmallVector<Operation *> affectedStatementsInLoop1 =
      nodeLoop1->findAffectedStatements(*(vstate.StatementInfoMap));

  // Statements inside the affected Statements2 must come inside the affected
  // statement1 statements
  SmallVector<Operation *> affectedStatementsInLoop2 =
      nodeLoop2->findAffectedStatements(*(vstate.StatementInfoMap));

  if (hasSmallerConstantRows(
          ((*(vstate.StatementInfoMap))[affectedStatementsInLoop2[0]]),
          ((*(vstate.StatementInfoMap))[affectedStatementsInLoop1[0]])))
    std::swap(affectedStatementsInLoop2, affectedStatementsInLoop1);

  int maxConstantInFusionDepth = maxConstantInDepthi(
      loop1Depth + 1 + fusionDepth, affectedStatementsInLoop1,
      *(vstate.StatementInfoMap));
  Operation *lastAffectedStatement = affectedStatementsInLoop2[0];
  Operation *firstAffectedStatement = affectedStatementsInLoop1[0];
  for (Operation *op : affectedStatementsInLoop2) {
    if (hasSmallerConstantRows(
            ((*(vstate.StatementInfoMap))[lastAffectedStatement]),
            ((*(vstate.StatementInfoMap))[op])))
      lastAffectedStatement = op;
  }
  SmallVector<int> lastAffectedConstantRowBeforeChange =
      ((*(vstate.StatementInfoMap))[lastAffectedStatement]).constantRows;

  for (unsigned i = 0; i < affectedStatementsInLoop2.size(); i++) {
    for (unsigned j = i; j < affectedStatementsInLoop2.size(); j++) {
      if (hasSmallerConstantRows(
              ((*(vstate.StatementInfoMap))[affectedStatementsInLoop2[j]]),
              ((*(vstate.StatementInfoMap))[affectedStatementsInLoop2[i]]))) {
        std::swap(affectedStatementsInLoop2[i], affectedStatementsInLoop2[j]);
      }
    }
  }
  for (Operation *op : affectedStatementsInLoop1) {
    (((*(vstate.StatementInfoMap))[op]).transformationList)
        .push_back(new LabelFusion());
  }
  maxConstantInFusionDepth++;
  // Update the constant rows inside the second input of the primitive
  for (Operation *op : affectedStatementsInLoop2) {

    SmallVector<int> &const1 = ((*(vstate.StatementInfoMap))[op]).constantRows;
    SmallVector<int> &const2 =
        ((*(vstate.StatementInfoMap))[firstAffectedStatement]).constantRows;
    (((*(vstate.StatementInfoMap))[op]).transformationList)
        .push_back(new LabelFusion());
    for (int i = 0; i <= fusionDepth; i++) {
      const1[loop1Depth + i] = const2[loop1Depth + i];
    }
    const1[loop1Depth + fusionDepth + 1] += maxConstantInFusionDepth;
  }

  // Fix statements after the fusion
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // any statement before the first statement in the distribution has no
    // effect
    if (!isControlFlow(op))
      if (hasSmallerConstantRows(
              ((*(vstate.StatementInfoMap))[lastAffectedStatement]),
              ((*(vstate.StatementInfoMap))[op]))) {
        // After the distribution set
        bool mustChange = true;
        SmallVector<int> &curStmtConstant =
            ((*(vstate.StatementInfoMap))[op]).constantRows;
        for (int i = 0; i < fusionDepth - 1; i++)
          if ((int)curStmtConstant.size() > i &&
              (int)lastAffectedConstantRowBeforeChange.size() > i &&
              curStmtConstant[i] != lastAffectedConstantRowBeforeChange[i])
            mustChange = false;
        if ((int)curStmtConstant.size() > fusionDepth && mustChange)
          // The fusionDepth is relative to the two target loops,
          // offset this by nesting depth of current op to get actual depth of
          // constants
          curStmtConstant[loop1Depth + fusionDepth]--;
      }

    // For debug only,
    // It will print the constant rows when we have
    // -debug
    ((*(vstate.StatementInfoMap))[op]).printConstantRow();
  });
  v1 = getResult();
  results.set(getResult().cast<OpResult>(), {target2});

  SmallVector<Node *> parents = {nodeLoop1, nodeLoop2};
  SmallVector<Node *> children;
  nodeLoop1->isLeaf = false;
  nodeLoop2->isLeaf = false;

  // building new nodes for the TG
  Node *tmp2 = new Node(loop1Depth, parents, children, NULL, true, false,
                        new LabelFusion());

  // set the TgMap to the value of the output
  (*(vstate.TgMap))[v1] = tmp2;
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ValidatorLoopTilingOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ValidatorLoopTilingOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);

  Value v1 = getTarget();
  auto payload1 = state.getPayloadOps(v1);
  AffineForOp loop = dyn_cast<AffineForOp>(*payload1.begin());
  int loopDepth = affine::getNestingDepth(loop);
  Operation *op1 = v1.getDefiningOp();
  Node *TgRoot = (vstate.TgRoot);
  func::FuncOp func = loop->getParentOfType<func::FuncOp>();

  // going into the new design, else is the previous design
  if (!func->hasAttr("PolyTool")) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  // We need to build nodes for the matchOp in their useCase since we don't
  // have them inside the validator
  if (dyn_cast_or_null<transform::MatchOp>(op1)) {
    SmallVector<Node *> parents = {TgRoot};
    SmallVector<Node *> children;
    Node *tmp = new Node(loopDepth, parents, children, loop, false, true,
                         new LabelMatch());
    (*(vstate.TgMap))[v1] = tmp;
  }

  // Nodes corresponding to the Transformation graph for the two loops in the
  // reorder operand
  Node *nodeLoop = (*(vstate.TgMap))[v1];
  loopDepth = nodeLoop->getDepth();
  SmallVector<int64_t, 4> tileSizes = extractFromI64ArrayAttr(getTileSizes());

  // finding the effected statements that must be tiled from the TG
  SmallVector<Operation *> affectedStatements =
      nodeLoop->findAffectedStatements(*(vstate.StatementInfoMap));

  // Checkpoint the scatterings and constantRows before being modified by
  // TileOp
  for (auto *op : affectedStatements) {
    if (isa<AffineYieldOp>(*op) || isa<AffineForOp>(*op))
      continue;
    affine::StatementInfo &info = (*(vstate.StatementInfoMap))[op];
    info.scatteringCheckPoint = info.scattering;
    info.constantRowsCheckPoint = info.constantRows;
  }

  for (auto op : affectedStatements) {
    LLVM_DEBUG(llvm::dbgs() << "Constant Rows: ");
    for (auto nei : ((*(vstate.StatementInfoMap))[op]).constantRows) {
      LLVM_DEBUG(llvm::dbgs() << nei << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }

  // First things that need to get fixed are the domains,
  // for example we have three iterators [i, j, k] and
  // we would like to tile [j, k]. The order of the columns
  // are [i, jT, kT, jj, kk]. On the other hand, if one wants
  // to tile [i, j] the order of columns are as follows:
  // [iT, jT, ii, jj, k]
  for (Operation *op : affectedStatements) {
    LLVM_DEBUG(llvm::dbgs() << "Operating on a affected statement\n");
    if (isa<AffineYieldOp>(*op) || isa<AffineForOp>(*op))
      continue;

    // This vector has booleans exactly until the maximum depth of the
    // statements
    SmallVector<bool> &isIteratorTiled =
        ((*(vstate.StatementInfoMap))[op]).isIteratorTiled;
    for (int i = 0; i < tileSizes.size(); i++) {
      if (tileSizes[i] != 1 && loopDepth + i < isIteratorTiled.size()) {
        isIteratorTiled[loopDepth + i] = true;
        ((*(vstate.StatementInfoMap))[op]).localTileSize++;
      }
    }
    SmallVector<SmallVector<int64_t>> &oldDomain =
        ((*(vstate.StatementInfoMap))[op]).domain;
    SmallVector<SmallVector<int64_t>> newDomain;
    SmallVector<int64_t> whereIsTileIterator;
    for (int i = 0; i < oldDomain.size(); i++) {
      SmallVector<int64_t> tmp;
      tmp.push_back(oldDomain[i][0]);
      int j = 1;
      while (j - 1 < isIteratorTiled.size() && !isIteratorTiled[j - 1]) {
        whereIsTileIterator.push_back(0);
        tmp.push_back(oldDomain[i][j]);
        j++;
      }
      int keepJ = j;
      while (j - 1 < isIteratorTiled.size() && isIteratorTiled[j - 1]) {
        tmp.push_back(0);
        j++;
      }
      for (int lineardependent = 0;
           lineardependent < ((*(vstate.StatementInfoMap))[op]).isFusedInto;
           lineardependent++) {
        tmp.push_back(0);
      }
      while (keepJ - 1 < isIteratorTiled.size() && isIteratorTiled[keepJ - 1]) {
        whereIsTileIterator.push_back(tmp.size());
        tmp.push_back(oldDomain[i][keepJ]);
        keepJ++;
      }
      while (j - 1 < isIteratorTiled.size() && !isIteratorTiled[j - 1]) {
        tmp.push_back(oldDomain[i][j]);
        j++;
      }
      for (j; j < oldDomain[i].size(); j++) {
        tmp.push_back(oldDomain[i][j]);
      }
      newDomain.push_back(tmp);
    }
    int temp = 0;
    for (int i = 0; i < tileSizes.size(); i++) {
      LLVM_DEBUG(llvm::dbgs()
                 << "isIteratorTiled.size()" << isIteratorTiled.size() << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "((*(vstate.StatementInfoMap))[op]).isFusedInto"
                 << ((*(vstate.StatementInfoMap))[op]).isFusedInto << "\n");
      if (isIteratorTiled.size() <= i &&
          ((*(vstate.StatementInfoMap))[op]).isFusedInto >= 1) {
        LLVM_DEBUG(llvm::dbgs() << "This is the linearly dependent column\n");
        SmallVector<int64_t> tmp1;
        SmallVector<int64_t> tmp2;
        for (int j = 0; j < newDomain[0].size(); j++) {
          tmp1.push_back(0);
          tmp2.push_back(0);
        }
        tmp1[0] = tmp2[0] = 1;
        tmp2.pop_back();
        tmp2.push_back(tileSizes[i] - 1);
        temp++;
        i -= temp;
        int ii = 0;
        for (int j = 2 * ((*(vstate.StatementInfoMap))[op]).maxRowCount + 2;
             j <
             (((*(vstate.StatementInfoMap))[op]).scattering[2 * i + 1]).size();
             j++) {
          if ((((*(vstate.StatementInfoMap))[op]).scattering[2 * i + 1])[j] ==
              -1)
            break;
          ii++;
        }
        ii += temp;
        tmp1[loopDepth + ii + 1] = -1 * tileSizes[i + temp];
        tmp2[loopDepth + ii + 1] = 1 * tileSizes[i + temp];
        tmp1[whereIsTileIterator[loopDepth + i]] = 1;
        tmp2[whereIsTileIterator[loopDepth + i]] = -1;
        newDomain.push_back(tmp1);
        newDomain.push_back(tmp2);
        i += temp;
        continue;
      }

      if (tileSizes[i] == 1 || isIteratorTiled.size() <= i)
        break;
      SmallVector<int64_t> tmp1;
      SmallVector<int64_t> tmp2;
      for (int j = 0; j < newDomain[0].size(); j++) {
        tmp1.push_back(0);
        tmp2.push_back(0);
      }
      tmp1[0] = tmp2[0] = 1;
      tmp2.pop_back();
      tmp2.push_back(tileSizes[i] - 1);

      // We want to take into account if any
      // reorder has happened before the tiling
      // for example if we have reorder the loop i
      // and j before tiling and then tile i an k
      // by tile size [1, 32, 32] so we need to be
      // aware that we need to change the domain for
      // (i and k) not (j and k) and ii is going to
      // do that for us
      int ii = 0;
      for (int j = 2 * ((*(vstate.StatementInfoMap))[op]).maxRowCount + 2;
           j <
           (((*(vstate.StatementInfoMap))[op]).scattering[2 * i + 1]).size();
           j++) {
        if ((((*(vstate.StatementInfoMap))[op]).scattering[2 * i + 1])[j] == -1)
          break;
        ii++;
      }
      tmp1[loopDepth + ii + 1] = -1 * tileSizes[i];
      tmp2[loopDepth + ii + 1] = 1 * tileSizes[i];
      tmp1[whereIsTileIterator[loopDepth + i]] = 1;
      tmp2[whereIsTileIterator[loopDepth + i]] = -1;
      newDomain.push_back(tmp1);
      newDomain.push_back(tmp2);
    }
    oldDomain = newDomain;
    LLVM_DEBUG(llvm::dbgs() << "Finished on a affected statement\n");
  }

  SmallVector<int> newConstants;
  SmallVector<int> previousOpConstants;
  int iteratorDomain = 0;
  for (int i = 0; i <= loopDepth; i++) {
    newConstants.push_back((
        ((*(vstate.StatementInfoMap))[affectedStatements[0]]).constantRows)[i]);
  }
  int outputCounter = 0;
  int currentIterator = 0;
  if (tileSizes[0] > 1) {
    v1 = getOperation()->getOpResult(0);
    SmallVector<Node *> parents = {nodeLoop};
    SmallVector<Node *> children;
    nodeLoop->isLeaf = false;
    Node *tmp1 = new Node(newConstants.size() - 1, parents, children, loop,
                          true, false, new LabelTile(tileSizes, newConstants));
    (*(vstate.TgMap))[v1] = tmp1;
  }

  // Fix constant rows
  previousOpConstants = newConstants;
  for (auto *op : affectedStatements) {
    SmallVector<int> &oldConstants =
        ((*(vstate.StatementInfoMap))[op]).constantRows;
    std::string str = reverseEngineerConstantRows(previousOpConstants,
                                                  oldConstants, loopDepth);
    if (str.size() == 0 || str.back() != 'S') {
      str.push_back('S');
    }
    LLVM_DEBUG(llvm::dbgs() << str << " the string \n");
    previousOpConstants = oldConstants;
    for (int i = 0; i < str.size(); i++) {
      LLVM_DEBUG(llvm::dbgs() << "constants: ");
      for (auto nei : newConstants) {
        LLVM_DEBUG(llvm::dbgs() << nei << " ");
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
      if (str[i] == 'F') {
        SmallVector<bool> &isIteratorTiled =
            ((*(vstate.StatementInfoMap))[op]).isIteratorTiled;

        // pattern match for the partial tile case:
        // for i
        //   S1
        //   for j
        //     S2
        // Tile i [32, 32]
        if (i - 1 >= 0 && str[i - 1] == 'S') {
          int depth = getNestingDepth(op);
          if (depth - 2 >= 0 &&
              (depth - 2 < isIteratorTiled.size() &&
               isIteratorTiled[depth - 2]) &&
              (depth - 1 < isIteratorTiled.size() &&
               isIteratorTiled[depth - 1])) {
            // both the "i" and the "j" are tiled
            // The partial tiled loop nest will look like
            // for i
            //   for ii
            //     S1
            //   for j
            //     for ii
            //       for jj
            newConstants.pop_back(); // Stmt col
            int incrementedConstant = newConstants.back() + 1;
            newConstants.pop_back();                     // old ii
            newConstants.push_back(incrementedConstant); // for j

            // at this point newConstants holds the prefix of the "for j" loop
            if (outputCounter < getOperation()->getResults().size()) {
              v1 = getOperation()->getOpResult(outputCounter);
              SmallVector<Node *> parents = {nodeLoop};
              SmallVector<Node *> children;
              nodeLoop->isLeaf = false;
              Node *tmp1 =
                  new Node(newConstants.size() - 1, parents, children, loop,
                           true, false, new LabelTile(tileSizes, newConstants));
              (*(vstate.TgMap))[v1] = tmp1;
              outputCounter++;
            }

            newConstants.push_back(0); // for ii
            // at this point newConstants holds the prefix of "for ii" loop
            if (outputCounter < getOperation()->getResults().size()) {
              v1 = getOperation()->getOpResult(outputCounter);
              SmallVector<Node *> parents = {nodeLoop};
              SmallVector<Node *> children;
              nodeLoop->isLeaf = false;
              Node *tmp1 =
                  new Node(newConstants.size() - 1, parents, children, loop,
                           true, false, new LabelTile(tileSizes, newConstants));
              (*(vstate.TgMap))[v1] = tmp1;
              outputCounter++;
            }

            newConstants.push_back(0); // for stmts under ii
            // another 0 will be added as jj later
            // when we look at the S after this F
          }
        } else {
          // newConstants.push_back(0); // for j (not considering tiling)

          if (outputCounter < getOperation()->getResults().size()) {
            v1 = getOperation()->getOpResult(outputCounter);
            SmallVector<Node *> parents = {nodeLoop};
            SmallVector<Node *> children;
            nodeLoop->isLeaf = false;
            Node *tmp1 =
                new Node(newConstants.size() - 1, parents, children, loop, true,
                         false, new LabelTile(tileSizes, newConstants));
            (*(vstate.TgMap))[v1] = tmp1;
          }
          newConstants.push_back(0); // for j (not considering tiling)
          outputCounter++;
        }

      } else if (str[i] == 'Y') {
        currentIterator = 0;
        newConstants.pop_back();
        for (int i = 0; i < iteratorDomain; i++)
          newConstants.pop_back();
        iteratorDomain = 0;
        int temp = newConstants.back() + 1;
        newConstants.pop_back();
        newConstants.push_back(temp);
      } else {
        SmallVector<bool> &isIteratorTiled =
            ((*(vstate.StatementInfoMap))[op]).isIteratorTiled;
        SmallVector<int> &oldConstants =
            ((*(vstate.StatementInfoMap))[op]).constantRows;
        // if it's the partial tiling case shown above, increment these values
        // later
        if (i + 1 >= str.size() || str[i + 1] != 'F')
          for (int i = currentIterator; i < isIteratorTiled.size(); i++) {
            currentIterator++;
            if (isIteratorTiled[i]) {
              iteratorDomain++;
              // newConstants.push_back(0);

              if (outputCounter < getOperation()->getResults().size()) {
                v1 = getOperation()->getOpResult(outputCounter);
                SmallVector<Node *> parents = {nodeLoop};
                SmallVector<Node *> children;
                nodeLoop->isLeaf = false;
                Node *tmp1 = new Node(newConstants.size() - 1, parents,
                                      children, loop, true, false,
                                      new LabelTile(tileSizes, newConstants));
                (*(vstate.TgMap))[v1] = tmp1;
              }
              newConstants.push_back(0);
              outputCounter++;
            }
          }
        oldConstants = newConstants;
        int temp = newConstants.back() + 1;
        newConstants.pop_back();
        ((*(vstate.StatementInfoMap))[op])
            .transformationList.push_back(
                new LabelTile(tileSizes, newConstants));
        newConstants.push_back(temp);
      }
    }
  }

  // Fix Scattering
  int maxRowCount = 0, maxColCount = 0;

  int maxTileSize = 0;

  // finds the max number of rows and cols accross the whole function
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if ((*(vstate.StatementInfoMap)).find(op) !=
        (*(vstate.StatementInfoMap)).end()) {
      int tilesize = ((*(vstate.StatementInfoMap))[op]).localTileSize;

      maxTileSize = std::max(tilesize, maxTileSize);

      SmallVector<SmallVector<int64_t>> a =
          ((*(vstate.StatementInfoMap))[op]).a;
      SmallVector<SmallVector<int64_t>> b =
          ((*(vstate.StatementInfoMap))[op]).b;
      SmallVector<SmallVector<int64_t>> c =
          ((*(vstate.StatementInfoMap))[op]).c;

      maxRowCount = std::max(a.size() + tilesize, (size_t)(maxRowCount));
      maxRowCount = std::max(b.size() + tilesize, (size_t)(maxRowCount));
      if (!a.empty())
        maxColCount = std::max(a[0].size() + tilesize, (size_t)(maxColCount));
      if (!b.empty())
        maxColCount = std::max(b[0].size() + tilesize, (size_t)(maxColCount));
    }
  });

  // Correct the maximum number of rows
  int numOrderingVar = maxRowCount + 1;
  LLVM_DEBUG(llvm::dbgs() << maxRowCount + numOrderingVar << "\n");
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if ((*(vstate.StatementInfoMap)).find(op) !=
        (*(vstate.StatementInfoMap)).end()) {
      int prevmaxRowCount = ((*(vstate.StatementInfoMap))[op]).maxRowCount;
      ((*(vstate.StatementInfoMap))[op]).maxRowCount = maxRowCount;
      ((*(vstate.StatementInfoMap))[op]).maxTileSize = maxTileSize;
      ((*(vstate.StatementInfoMap))[op]).maxColCount = maxColCount;
      SmallVector<SmallVector<int64_t>> &oldScattering =
          ((*(vstate.StatementInfoMap))[op]).scattering;
      int temp = oldScattering.size();
      for (int j = 0; j < oldScattering.size(); j++) {
        SmallVector<int64_t> tmp;
        int k = 0;
        for (k; k <= 2 * prevmaxRowCount + 1; k++) {
          tmp.push_back(oldScattering[j][k]);
        }
        for (int i = temp; i <= maxRowCount + maxRowCount; i++) {
          tmp.push_back(0);
        }
        for (k; k < oldScattering[j].size(); k++) {
          tmp.push_back(oldScattering[j][k]);
        }
        oldScattering[j] = tmp;
      }
      for (int i = temp; i <= maxRowCount + maxRowCount; i++) {
        SmallVector<int64_t> tmp;
        if (oldScattering.size()) {
          for (int j = 0; j < oldScattering[0].size(); j++) {
            if (j == i + 1)
              tmp.push_back(1);
            else
              tmp.push_back(0);
          }
          oldScattering.push_back(tmp);
        }
      }
    }
  });
  // for the loop which has been tiled, we need to insert the tiled iterators
  for (Operation *op : affectedStatements) {
    SmallVector<bool> &isIteratorTiled =
        ((*(vstate.StatementInfoMap))[op]).isIteratorTiled;
    int startPos = 2 * maxRowCount + 2;
    int i = 0;
    int cnt = 0; // number of tile iterators which must be added
    while (i < tileSizes.size() &&
           (tileSizes[i] == 1 || loopDepth + i >= isIteratorTiled.size())) {
      startPos++;
      i++;
    }
    while (i < tileSizes.size() && tileSizes[i] != 1 &&
           loopDepth + i < isIteratorTiled.size()) {
      startPos++;
      cnt++;
      i++;
    }
    int lastOneWrittenInRow = -1;
    SmallVector<SmallVector<int64_t>> &oldScattering =
        ((*(vstate.StatementInfoMap))[op]).scattering;
    for (int k = 0; k < cnt; k++) {
      for (int j = 0; j < oldScattering.size(); j++) {
        auto *it = oldScattering[j].begin() + startPos + k;
        if (j - 2 * cnt >= 0 &&
            oldScattering[j - 2 * cnt][2 * maxRowCount + 2 + k] == -1) {
          oldScattering[j].insert(it, -1);
          lastOneWrittenInRow = j;
        } else
          oldScattering[j].insert(it, 0);
      }
    }
    for (int k = cnt; k < isIteratorTiled.size(); k++) {
      oldScattering[2 * k + 1][startPos + k] = 0;
      oldScattering[2 * cnt + 2 * k + 1][startPos + k] = -1;
    }
    // if we have a linear dependent column one more column must be added
    // I will set that column all equal to zero since the correction will
    // fix it anyway
    for (int lineardependent = 0;
         lineardependent < ((*(vstate.StatementInfoMap))[op]).isFusedInto;
         lineardependent++) {
      lastOneWrittenInRow += 2;
      for (int j = 0; j < oldScattering.size(); j++) {
        long *it = oldScattering[j].begin() + startPos + cnt + lineardependent;
        if (j == lastOneWrittenInRow)
          oldScattering[j].insert(it, -1);
        else
          oldScattering[j].insert(it, 0);
      }
    }
  }
  // print the results for debuging
  func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    ((*(vstate.StatementInfoMap))[op]).printConstantRow();
    ((*(vstate.StatementInfoMap))[op]).printDomain();
    ((*(vstate.StatementInfoMap))[op]).printScattering();
  });

  for (uint64_t i = 0; i < getOperation()->getResults().size(); i++) {
    results.set(getOperation()->getOpResult(i), {loop});
  }
  LLVM_DEBUG(llvm::dbgs() << "Validator TileOp succesfully returned\n ");
  return DiagnosedSilenceableFailure::success();
}

void transform::ValidatorLoopTilingOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : ";
  p.printFunctionalType(TypeRange(getOperand().getType()),
                        getResults().getTypes());
}

ParseResult transform::ValidatorLoopTilingOp::parse(OpAsmParser &parser,
                                                    OperationState &result) {
  OpAsmParser::UnresolvedOperand targetOperand;
  if (parser.parseOperand(targetOperand) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  FunctionType trailingType;
  SMLoc typeLoc;
  if (parser.getCurrentLocation(&typeLoc) ||
      parser.parseColonType(trailingType)) {
    return failure();
  }
  if (trailingType.getNumInputs() != 1)
    return parser.emitError(typeLoc) << "expected one input type";

  result.addTypes(trailingType.getResults());
  if (parser.resolveOperand(targetOperand, trailingType.getInput(0),
                            result.operands))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// ValidatorBlockReorderOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure transform::ValidatorBlockReorderOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);

  SmallVector<int64_t> permutationVec =
      extractFromIntegerArrayAttr<int64_t>(getPermutation());
  assert(getInputs().size() == getPermutation().size() &&
         "number of inputs expected a single target op");

  int minPos = permutationVec[0]; // Find smallest position
  for (uint64_t i = 0; i < getPermutation().size(); i++)
    minPos = minPos < permutationVec[i] ? minPos : permutationVec[i];
  LLVM_DEBUG(llvm::dbgs() << "minPos: " << minPos << "\n");
  for (uint64_t i = 0; i < getPermutation().size(); i++) {
    // Find the ith operand position in permutationVec
    int pos = -1;
    for (uint64_t j = 0; j < getPermutation().size(); j++) {
      if ((permutationVec[j] - minPos) == i)
        pos = j;
    }
    assert(pos != -1 && "expect to find a proper position");
    Value operand = getOperand(pos);
    auto payload = state.getPayloadOps(getOperand(pos));

    // TODO: inputs may not need to be AffineForOp. All blocks must be at the
    // same level of scoping before entering BlockReorder
    AffineForOp loop = dyn_cast<AffineForOp>(*payload.begin());
    int loopDepth = affine::getNestingDepth(loop);

    // If any of the BlockReorder inputs are from MatchOp, create a TG node
    if (dyn_cast_or_null<transform::MatchOp>(operand.getDefiningOp())) {
      SmallVector<Node *> parents = {vstate.TgRoot};
      SmallVector<Node *> children;
      (*(vstate.TgMap))[operand] = new Node(loopDepth, parents, children, loop,
                                            false, true, new LabelMatch());
    }
    Node *nodeCurr = (*(vstate.TgMap))[operand];
    SmallVector<Operation *> affectedStatements =
        nodeCurr->findAffectedStatements(*(vstate.StatementInfoMap));
    int scope = loopDepth; // Only support scope 0 with no error checking.
    for (auto op : affectedStatements) {
      affine::StatementInfo &info = (*(vstate.StatementInfoMap))[op];
      info.constantRows[scope] = permutationVec[pos];
      info.printConstantRow(false);
    }

    Value output = getOperation()->getOpResult(i);
    Value input = getOperand(i);
    Node *nodeLoop = (*(vstate.TgMap))[input];
    SmallVector<Node *> parents = {nodeLoop};
    SmallVector<Node *> children;
    nodeLoop->isLeaf = false;
    Node *tmp =
        new Node(nodeLoop->getDepth(), parents, children, loop, true, false,
                 new LabelBlockReorder(scope, nodeLoop->getDepth(),
                                       nodeCurr->getDepth()));
    (*(vstate.TgMap))[output] = tmp;
    results.set(getOperation()->getOpResult(i).cast<OpResult>(), {loop});
  }
  LLVM_DEBUG(llvm::dbgs() << "Validator BlockReorderOp succesfully returned\n");
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ValidatorSkewOp
//===----------------------------------------------------------------------===//
DiagnosedSilenceableFailure
transform::ValidatorSkewOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {

  LLVM_DEBUG(llvm::dbgs() << "SkewOp\n");
  ValidatorTransformState &vstate =
      static_cast<ValidatorTransformState &>(state);

  // Get references to the two loops
  Value vOuter = getOuter();
  Value vInner = getInner();

  // Get actual for loop objects
  AffineForOp loopOuter =
      dyn_cast<AffineForOp>(*state.getPayloadOps(vOuter).begin());
  AffineForOp loopInner =
      dyn_cast<AffineForOp>(*state.getPayloadOps(vInner).begin());

  if (!loopOuter->getParentOfType<func::FuncOp>()->hasAttr("PolyTool")) {
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  // Get the depths of the two loops
  int OuterDepth = affine::getNestingDepth(loopOuter);
  int InnerDepth = affine::getNestingDepth(loopInner);

  // If the loop we are looking at is the result of some
  // previous operation on loops and that operation is a MatchOp.
  // This means we obtained this loop using transform.structured.match
  // As a result, we need to allocate a node in the TG map to represent it
  if (dyn_cast_or_null<transform::MatchOp>(vOuter.getDefiningOp())) {
    SmallVector<Node *> parents = {vstate.TgRoot};
    SmallVector<Node *> children;

    (*(vstate.TgMap))[vOuter] =
        new Node(OuterDepth, parents, children, loopOuter, false, true,
                 new LabelMatch());
  }
  if (dyn_cast_or_null<transform::MatchOp>(vInner.getDefiningOp())) {
    SmallVector<Node *> parents = {vstate.TgRoot};
    SmallVector<Node *> children;

    (*(vstate.TgMap))[vInner] =
        new Node(InnerDepth, parents, children, loopInner, false, true,
                 new LabelMatch());
  }

  Node *nodeOuter = (*(vstate.TgMap))[vOuter];
  Node *nodeInner = (*(vstate.TgMap))[vInner];

  // Yes this dosen't actually change their values,
  // but whatever its fine (copying this from old code)
  OuterDepth = nodeOuter->getDepth();
  InnerDepth = nodeInner->getDepth();

  // Despite having outter and inner be the names of the loops,
  // it sometimes makes sense to have skews go the other way around
  // As a result, use the ternary operator to ensure this works
  SmallVector<Operation *> affectedStatements =
      (OuterDepth < InnerDepth ? nodeOuter : nodeInner)
          ->findAffectedStatements(*(vstate.StatementInfoMap));

  LLVM_DEBUG(llvm::dbgs() << "SKEW OP ON " << affectedStatements.size()
                          << " STATEMENTS IN PROCESS...\n");

  for (auto op : affectedStatements) {
    ((*(vstate.StatementInfoMap))[op])
        .LoopSkew(InnerDepth, OuterDepth, getAmount());
  }

  Value vSkewed = getOperation()->getOpResult(0);
  Value vUnchanged = getOperation()->getOpResult(1);

  // We add the new node we created into the TgMap
  nodeOuter->isLeaf = false;
  nodeInner->isLeaf = false;

  SmallVector<Node *> parents = {nodeOuter, nodeInner};
  SmallVector<Node *> children;
  (*(vstate.TgMap))[vSkewed] =
      new Node(OuterDepth, parents, children, loopOuter, true, false,
               new LabelSkew(OuterDepth, InnerDepth, getAmount()));

  parents = {nodeInner, nodeOuter};
  children = {};
  (*(vstate.TgMap))[vUnchanged] =
      new Node(InnerDepth, parents, children, loopInner, true, false,
               new LabelSkew(OuterDepth, InnerDepth, getAmount()));

  results.set(getOperation()->getOpResult(0).cast<OpResult>(), {loopOuter});
  results.set(getOperation()->getOpResult(1).cast<OpResult>(), {loopInner});
  return DiagnosedSilenceableFailure::success();
}

namespace {
class ValidatorTransformDialectExtension
    : public transform::TransformDialectExtension<
          ValidatorTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    // declareGeneratedDialect<AffineDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Schedule/TransformOps/ValidatorTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Schedule/TransformOps/ValidatorTransformOps.cpp.inc"

void mlir::validator::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<ValidatorTransformDialectExtension>();
}
