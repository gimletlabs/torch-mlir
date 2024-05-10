//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class RaiseIfOpTypes : public OpRewritePattern<PrimIfOp> {

public:
  using OpRewritePattern::OpRewritePattern;

  Value getEffectiveOperand(Value operand) const {
    if (auto cast = operand.getDefiningOp<TensorStaticInfoCastOp>()) {
      return cast.getOperand();
    }
    return operand;
  }

  LogicalResult matchAndRewrite(PrimIfOp ifOp,
                                PatternRewriter &rewriter) const override {
    auto thenYield = llvm::dyn_cast<PrimIfYieldOp>(ifOp.getThenRegion().front().getTerminator());
    if (!thenYield) {
      return failure();
    }
    auto elseYield = llvm::dyn_cast<PrimIfYieldOp>(ifOp.getElseRegion().front().getTerminator());
    if (!elseYield) {
      return failure();
    }
    if (thenYield.getNumOperands() != elseYield.getNumOperands()) {
      return failure();
    }

    bool allTypesSame = true;
    SmallVector<Value> newThenOperands;
    SmallVector<Value> newElseOperands;
    SmallVector<Type> newResultTypes;
    for (size_t i = 0; i < thenYield.getNumOperands(); ++i) {
      auto thenOperand = getEffectiveOperand(thenYield.getOperand(i));
      auto elseOperand = getEffectiveOperand(elseYield.getOperand(i));
      if (thenOperand.getType() != elseOperand.getType()) {
        return failure();
      }
      if (thenOperand.getType() != ifOp.getResult(i).getType()) {
        allTypesSame = false;
      }
      newResultTypes.push_back(thenOperand.getType());
      newThenOperands.push_back(thenOperand);
      newElseOperands.push_back(elseOperand);
    }
    if (allTypesSame) {
      return failure();
    }

    rewriter.startOpModification(thenYield);
    thenYield->setOperands(newThenOperands);
    rewriter.finalizeOpModification(thenYield);

    rewriter.startOpModification(elseYield);
    elseYield->setOperands(newElseOperands);
    rewriter.finalizeOpModification(elseYield);

    rewriter.startOpModification(ifOp);
    rewriter.setInsertionPointAfter(ifOp);
    for (auto result : llvm::enumerate(ifOp.getResults())) {
      auto newType = newResultTypes[result.index()];
      if (!llvm::isa<BaseTensorType>(newType)) {
        continue;
      }
      auto oldType = result.value().getType();
      result.value().setType(newType);
      if (result.value().getType() == oldType) {
        continue;
      }
      auto newResult =
          copyTensorToType(rewriter, ifOp.getLoc(),
                           oldType.cast<BaseTensorType>(), result.value());
      rewriter.replaceAllUsesExcept(result.value(), newResult,
                                    newResult.getDefiningOp());
    }
    rewriter.finalizeOpModification(ifOp);

    return success();
  }
};

class RaiseIfOpShapesDtypesPass
    : public RaiseIfOpShapesDtypesBase<RaiseIfOpShapesDtypesPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto func = getOperation();

    RewritePatternSet patterns(context);
    patterns.insert<RaiseIfOpTypes>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createRaiseIfOpShapesDtypesPass() {
  return std::make_unique<RaiseIfOpShapesDtypesPass>();
}
