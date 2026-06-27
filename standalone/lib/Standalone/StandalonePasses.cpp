//===- StandalonePasses.cpp - Standalone passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#define GEN_PASS_DEF_LOWERREQUANTIZETOFUNCCALL
#define GEN_PASS_DEF_CREATECINTERFACEENTRYWRAPPERS
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

static bool is8x8IntegerMemRef(Type type, unsigned width) {
  auto memrefType = dyn_cast<MemRefType>(type);
  return memrefType && memrefType.getRank() == 2 &&
         memrefType.getDimSize(0) == 8 &&
         memrefType.getDimSize(1) == 8 &&
         memrefType.getElementType().isInteger(width);
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

    if (!is8x8IntegerMemRef(lhs.getType(), 8) ||
        !is8x8IntegerMemRef(rhs.getType(), 8) ||
        !is8x8IntegerMemRef(acc.getType(), 32))
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

static bool isMatmulFunctionToWrap(func::FuncOp func) {
  if (func.getSymName() != "matmul")
    return false;

  if (func.isPrivate() || func.isDeclaration())
    return false;

  if (func.getSymName().ends_with("_entry"))
    return false;

  FunctionType functionType = func.getFunctionType();
  if (functionType.getNumInputs() != 2 || functionType.getNumResults() != 1)
    return false;

  if (!isa<MemRefType>(functionType.getResult(0)))
    return false;

  return llvm::all_of(functionType.getInputs(),
                      [](Type type) { return isa<MemRefType>(type); });
}

class CreateCInterfaceEntryWrappers
    : public impl::CreateCInterfaceEntryWrappersBase<
          CreateCInterfaceEntryWrappers> {
public:
  using impl::CreateCInterfaceEntryWrappersBase<
      CreateCInterfaceEntryWrappers>::CreateCInterfaceEntryWrappersBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();
    OpBuilder builder(context);

    SmallVector<func::FuncOp> funcsToWrap;
    module.walk([&](func::FuncOp func) {
      if (isMatmulFunctionToWrap(func))
        funcsToWrap.push_back(func);
    });

    for (func::FuncOp func : funcsToWrap) {
      std::string entryName = (func.getSymName() + "_entry").str();
      if (module.lookupSymbol<func::FuncOp>(entryName))
        continue;

      FunctionType functionType = func.getFunctionType();
      SmallVector<Type> entryInputs(functionType.getInputs().begin(),
                                    functionType.getInputs().end());
      entryInputs.push_back(functionType.getResult(0));
      auto entryType = builder.getFunctionType(entryInputs, {});

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(func);
      auto entry = func::FuncOp::create(builder, func.getLoc(), entryName,
                                        entryType);
      entry->setAttr("llvm.emit_c_interface", builder.getUnitAttr());

      Block *body = entry.addEntryBlock();
      builder.setInsertionPointToStart(body);

      ValueRange originalInputs = body->getArguments().drop_back();
      Value output = body->getArguments().back();
      auto call = func::CallOp::create(builder, func.getLoc(),
                                       func.getSymName(),
                                       functionType.getResults(),
                                       originalInputs);

      memref::CopyOp::create(builder, func.getLoc(), call.getResult(0),
                             output);
      func::ReturnOp::create(builder, func.getLoc());
    }
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

// Framework helper: replace an accelerator op (no results) with a func.call to
// runtime function `symbol`, declaring the private function on first use.
// Reuse this from per-IP lowering passes so their pattern body stays tiny.
static void lowerToRuntimeCall(Operation *op, StringRef symbol,
                              ValueRange operands, PatternRewriter &rewriter) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (!module.lookupSymbol<func::FuncOp>(symbol)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto fnType = rewriter.getFunctionType(operands.getTypes(), {});
    auto fn = func::FuncOp::create(rewriter, op->getLoc(), symbol, fnType);
    fn.setPrivate();
  }
  rewriter.replaceOpWithNewOp<func::CallOp>(op, symbol, TypeRange{}, operands);
}

class LowerRequantizeToCall
    : public OpRewritePattern<standalone::RequantizeOp> {
public:
  using OpRewritePattern<standalone::RequantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(standalone::RequantizeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value mult = arith::ConstantOp::create(rewriter, loc, op.getMultAttr());
    Value shift = arith::ConstantOp::create(rewriter, loc, op.getShiftAttr());
    Value zeroPoint =
        arith::ConstantOp::create(rewriter, loc, op.getZeroPointAttr());

    lowerToRuntimeCall(
        op, "epilogue_8x8",
        ValueRange{op.getAcc(), op.getOut(), mult, shift, zeroPoint}, rewriter);

    return success();
  }
};

class LowerRequantizeToFuncCall
    : public impl::LowerRequantizeToFuncCallBase<LowerRequantizeToFuncCall> {
public:
  using impl::LowerRequantizeToFuncCallBase<
      LowerRequantizeToFuncCall>::LowerRequantizeToFuncCallBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerRequantizeToCall>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::standalone
