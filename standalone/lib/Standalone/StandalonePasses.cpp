//===- StandalonePasses.cpp - Standalone passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/StandalonePasses.h"

namespace mlir::standalone {
#define GEN_PASS_DEF_STANDALONESWITCHBARFOO
#define GEN_PASS_DEF_CONVERTLINALGMATMULTOSYSTOLIC
#define GEN_PASS_DEF_LOWERSYSTOLICTOFUNCCALL
#include "Standalone/StandalonePasses.h.inc"

namespace {
class StandaloneSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class StandaloneSwitchBarFoo
    : public impl::StandaloneSwitchBarFooBase<StandaloneSwitchBarFoo> {
public:
  using impl::StandaloneSwitchBarFooBase<
      StandaloneSwitchBarFoo>::StandaloneSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<StandaloneSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

static bool is8x8F32MemRef(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  return memrefType && memrefType.getRank() == 2 &&
         memrefType.getDimSize(0) == 8 &&
         memrefType.getDimSize(1) == 8 &&
         memrefType.getElementType().isF32();
}

class ConvertMatmulToSystolicPattern
    : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    Value acc = op.getOutputs()[0];

    if (!is8x8F32MemRef(lhs.getType()) ||
        !is8x8F32MemRef(rhs.getType()) ||
        !is8x8F32MemRef(acc.getType()))
      return failure();

    rewriter.replaceOpWithNewOp<standalone::SystolicMatmulOp>(
        op, lhs, rhs, acc);

    return success();
  }
};

class ConvertLinalgMatmulToSystolic
    : public impl::ConvertLinalgMatmulToSystolicBase<
          ConvertLinalgMatmulToSystolic> {
public:
  using impl::ConvertLinalgMatmulToSystolicBase<
      ConvertLinalgMatmulToSystolic>::ConvertLinalgMatmulToSystolicBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandaloneDialect>();
  }

  void runOnOperation() final {
    getContext().loadDialect<StandaloneDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertMatmulToSystolicPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

class LowerSystolicMatmulToCall
    : public OpRewritePattern<standalone::SystolicMatmulOp> {
public:
  using OpRewritePattern<standalone::SystolicMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(standalone::SystolicMatmulOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value acc = op.getAcc();

    auto fn = module.lookupSymbol<func::FuncOp>("systolic_matmul_8x8");
    if (!fn) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto fnType = rewriter.getFunctionType(
          {lhs.getType(), rhs.getType(), acc.getType()}, {});

      fn = func::FuncOp::create(
          rewriter, op.getLoc(), "systolic_matmul_8x8", fnType);
      fn.setPrivate();
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(
        op,
        "systolic_matmul_8x8",
        TypeRange{},
        ValueRange{lhs, rhs, acc});

    return success();
  }
};

class LowerSystolicToFuncCall
    : public impl::LowerSystolicToFuncCallBase<LowerSystolicToFuncCall> {
public:
  using impl::LowerSystolicToFuncCallBase<
      LowerSystolicToFuncCall>::LowerSystolicToFuncCallBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerSystolicMatmulToCall>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::standalone
