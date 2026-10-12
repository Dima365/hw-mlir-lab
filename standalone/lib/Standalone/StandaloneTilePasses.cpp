//===- StandaloneTilePasses.cpp - Accelerator tile lowering ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/StandalonePasses.h"

namespace mlir::standalone {
#define GEN_PASS_DEF_PREPAREU8S8MATMUL
#define GEN_PASS_DEF_LOWERCONVREQUANTTOTILES
#define GEN_PASS_DEF_CONVERTLINALGU8S8MATMULTOSYSTOLIC
#define GEN_PASS_DEF_LOWERQADDRELUTOTILES
#include "Standalone/StandalonePasses.h.inc"

namespace {
constexpr StringLiteral kU8S8MatmulMarker = "standalone.u8s8_matmul";

static Value constantIndex(OpBuilder &builder, Location loc, int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

static Value castIntegerSignedness(OpBuilder &builder, Location loc,
                                   Value value, Type targetType) {
  if (value.getType() == targetType)
    return value;
  return UnrealizedConversionCastOp::create(builder, loc, TypeRange{targetType},
                                            value)
      .getResult(0);
}

static SmallVector<AffineMap> getU8S8MatmulIndexingMaps(MLIRContext *context) {
  AffineExpr m = getAffineDimExpr(0, context);
  AffineExpr n = getAffineDimExpr(1, context);
  AffineExpr k = getAffineDimExpr(2, context);
  return {
      AffineMap::get(3, 0, {m, k}, context),
      AffineMap::get(3, 0, {k, n}, context),
      AffineMap::get(3, 0, context),
      AffineMap::get(3, 0, {m, n}, context),
  };
}

static SmallVector<utils::IteratorType> getU8S8MatmulIteratorTypes() {
  return {
      utils::IteratorType::parallel,
      utils::IteratorType::parallel,
      utils::IteratorType::reduction,
  };
}

static IntegerType get8x8IntegerElementType(Value value, unsigned width) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getShape() != ArrayRef<int64_t>({8, 8}))
    return {};
  auto elementType = dyn_cast<IntegerType>(type.getElementType());
  if (!elementType || elementType.getWidth() != width)
    return {};
  return elementType;
}

static bool hasU8S8MatmulBody(linalg::GenericOp op, Value &zeroPoint) {
  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 4 && body.getNumArguments() != 3)
    return false;

  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield->getNumOperands() != 1)
    return false;
  if (llvm::any_of(body.without_terminator(),
                   [](Operation &op) { return !isMemoryEffectFree(&op); }))
    return false;
  auto add = yield->getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  Value product;
  Value acc = body.getArguments().back();
  if (add->getOperand(0) == acc)
    product = add->getOperand(1);
  else if (add->getOperand(1) == acc)
    product = add->getOperand(0);
  else
    return false;

  auto multiply = product.getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return false;

  Value lhs;
  arith::ExtSIOp rhsWide;
  for (Value operand : multiply->getOperands()) {
    if (operand.getDefiningOp<arith::SubIOp>() ||
        operand.getDefiningOp<arith::ExtUIOp>())
      lhs = operand;
    else if (auto candidate = operand.getDefiningOp<arith::ExtSIOp>())
      rhsWide = candidate;
  }
  if (!lhs || !rhsWide || rhsWide->getOperand(0) != body.getArgument(1))
    return false;
  zeroPoint = {};
  if (auto subtract = lhs.getDefiningOp<arith::SubIOp>()) {
    zeroPoint = subtract.getRhs();
    lhs = subtract.getLhs();
    if (body.getNumArguments() == 4) {
      if (zeroPoint != body.getArgument(2))
        return false;
      zeroPoint = op.getInputs()[2];
    }
  } else if (body.getNumArguments() == 4) {
    return false;
  }
  auto lhsWide = lhs.getDefiningOp<arith::ExtUIOp>();
  if (!lhsWide)
    return false;
  if (lhsWide.getIn() == body.getArgument(0))
    return true;
  auto lhsCast =
      lhsWide->getOperand(0).getDefiningOp<UnrealizedConversionCastOp>();
  return lhsCast && lhsCast->getNumOperands() == 1 &&
         lhsCast->getNumResults() == 1 &&
         lhsCast->getOperand(0) == body.getArgument(0) &&
         lhsCast->getResult(0).getType().isSignlessInteger(8);
}

class ConvertLinalgU8S8MatmulPattern
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr(kU8S8MatmulMarker))
      return failure();
    if ((op.getInputs().size() != 3 && op.getInputs().size() != 2) ||
        op.getOutputs().size() != 1 || op.getNumResults() != 0)
      return rewriter.notifyMatchFailure(
          op, "expected two or three inputs, one output and buffer semantics");

    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    Value acc = op.getOutputs()[0];
    IntegerType lhsElementType = get8x8IntegerElementType(lhs, 8);
    IntegerType rhsElementType = get8x8IntegerElementType(rhs, 8);
    IntegerType accElementType = get8x8IntegerElementType(acc, 32);
    if (!lhsElementType || lhsElementType.isSigned() || !rhsElementType ||
        !rhsElementType.isSignless() || !accElementType ||
        !accElementType.isSignless())
      return rewriter.notifyMatchFailure(
          op, "expected 8x8 i8/ui8, i8 and i32 buffers");
    auto maps = getU8S8MatmulIndexingMaps(rewriter.getContext());
    if (op.getInputs().size() == 2)
      maps.erase(maps.begin() + 2);
    if (op.getIndexingMapsArray() != maps ||
        op.getIteratorTypesArray() != getU8S8MatmulIteratorTypes())
      return rewriter.notifyMatchFailure(
          op, "expected matmul indexing maps and iterator types");
    Value lhsZeroPoint;
    if (!hasU8S8MatmulBody(op, lhsZeroPoint))
      return rewriter.notifyMatchFailure(
          op, "expected exact u8s8 zero-point-corrected matmul arithmetic");

    APInt zeroPoint(32, 0);
    if (lhsZeroPoint &&
        (!matchPattern(lhsZeroPoint, m_ConstantInt(&zeroPoint)) ||
         zeroPoint.getSExtValue() < 0 || zeroPoint.getSExtValue() > 255))
      return rewriter.notifyMatchFailure(
          op, "lhs zero point must be a constant in [0, 255]");

    rewriter.replaceOpWithNewOp<standalone::SystolicU8S8MatmulOp>(
        op, lhs, rhs, acc,
        rewriter.getI32IntegerAttr(zeroPoint.getSExtValue()));
    return success();
  }
};

class ConvertLinalgU8S8MatmulToSystolic
    : public impl::ConvertLinalgU8S8MatmulToSystolicBase<
          ConvertLinalgU8S8MatmulToSystolic> {
public:
  using impl::ConvertLinalgU8S8MatmulToSystolicBase<
      ConvertLinalgU8S8MatmulToSystolic>::ConvertLinalgU8S8MatmulToSystolicBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertLinalgU8S8MatmulPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    bool hasUnconvertedOp = false;
    getOperation().walk([&](linalg::GenericOp op) {
      if (!op->hasAttr(kU8S8MatmulMarker))
        return;
      op.emitError("marked u8s8 matmul was not converted to the systolic op");
      hasUnconvertedOp = true;
    });
    if (hasUnconvertedOp)
      signalPassFailure();
  }
};

// The fork produces input[B,K,P] * weight[F,K] -> output[B,F,P].
// Normalize storage with standard transpose/reshape ops; im2col itself remains
// the upstream library transformation, not a second implementation here.
static Value transposeTensor(OpBuilder &builder, Location loc, Value input,
                             ArrayRef<int64_t> permutation) {
  auto type = cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> shape;
  for (int64_t dim : permutation)
    shape.push_back(type.getDimSize(dim));
  Value empty =
      tensor::EmptyOp::create(builder, loc, shape, type.getElementType());
  return linalg::TransposeOp::create(builder, loc, input, empty, permutation)
      .getResult()[0];
}

static SmallVector<AffineMap> getIm2ColMatmulMaps(MLIRContext *context) {
  AffineExpr b, f, p, k;
  bindDims(context, b, f, p, k);
  return {AffineMap::get(4, 0, {b, k, p}, context),
          AffineMap::get(4, 0, {f, k}, context), AffineMap::get(4, 0, context),
          AffineMap::get(4, 0, context),
          AffineMap::get(4, 0, {b, f, p}, context)};
}

static bool isExtendedInput(Value value, Value input, bool isUnsigned) {
  Operation *extension = value.getDefiningOp();
  if (!extension || extension->getNumOperands() != 1 ||
      !value.getType().isSignlessInteger(32))
    return false;
  if (isUnsigned ? !isa<arith::ExtUIOp>(extension)
                 : !isa<arith::ExtSIOp>(extension))
    return false;
  Value narrow = extension->getOperand(0);
  if (auto cast = narrow.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
        !narrow.getType().isSignlessInteger(8) ||
        !cast->getOperand(0).getType().isUnsignedInteger(8))
      return false;
    narrow = cast->getOperand(0);
  }
  return narrow == input;
}

static bool isCenteredInput(Value value, Value input, Value zp,
                            bool isUnsigned) {
  auto sub = value.getDefiningOp<arith::SubIOp>();
  return sub && sub.getRhs() == zp &&
         isExtendedInput(sub.getLhs(), input, isUnsigned);
}

static bool hasQuantizedConvBody(linalg::GenericOp op) {
  if (op.getInputs().size() != 4 || op.getOutputs().size() != 1 ||
      op.getNumResults() != 1 || !op.hasPureTensorSemantics() ||
      op.getIteratorTypesArray() !=
          SmallVector<utils::IteratorType>{
              utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::parallel, utils::IteratorType::reduction})
    return false;
  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 5 ||
      llvm::any_of(body.without_terminator(),
                   [](Operation &op) { return !isMemoryEffectFree(&op); }))
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;
  auto add = yield.getValues()[0].getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;
  Value product = add.getLhs() == body.getArgument(4)   ? add.getRhs()
                  : add.getRhs() == body.getArgument(4) ? add.getLhs()
                                                        : Value{};
  auto mul = product ? product.getDefiningOp<arith::MulIOp>() : arith::MulIOp{};
  if (!mul)
    return false;
  auto matches = [&](Value a, Value w) {
    return isCenteredInput(a, body.getArgument(0), body.getArgument(2), true) &&
           isCenteredInput(w, body.getArgument(1), body.getArgument(3), false);
  };
  return matches(mul.getLhs(), mul.getRhs()) ||
         matches(mul.getRhs(), mul.getLhs());
}

class PrepareU8S8Matmul
    : public impl::PrepareU8S8MatmulBase<PrepareU8S8Matmul> {
public:
  using impl::PrepareU8S8MatmulBase<PrepareU8S8Matmul>::PrepareU8S8MatmulBase;

  void runOnOperation() final {
    IRRewriter rewriter(&getContext());
    SmallVector<linalg::GenericOp> candidates;
    getOperation().walk([&](linalg::GenericOp op) {
      if (op.getIndexingMapsArray() == getIm2ColMatmulMaps(&getContext()))
        candidates.push_back(op);
    });
    for (auto op : candidates) {
      if (failed(prepare(op, rewriter))) {
        signalPassFailure();
        return;
      }
    }
    // Do not silently leave a convolution on the CPU if the fork conversion
    // could not handle its shape or tensor semantics.
    getOperation().walk([&](linalg::Conv2DNchwFchwQOp op) {
      op.emitError("quantized convolution was not converted by im2col");
      signalPassFailure();
    });
  }

private:
  LogicalResult prepare(linalg::GenericOp op, IRRewriter &rewriter) {
    if (!hasQuantizedConvBody(op))
      return op.emitError("expected unsigned activation / signed weight "
                          "zero-point-corrected im2col arithmetic");
    APInt inputZp, weightZp;
    if (!matchPattern(op.getInputs()[2], m_ConstantInt(&inputZp)) ||
        inputZp.getSExtValue() < 0 || inputZp.getSExtValue() > 255)
      return op.emitError("activation zero point must be constant in [0, 255]");
    if (!matchPattern(op.getInputs()[3], m_ConstantInt(&weightZp)) ||
        !weightZp.isZero())
      return op.emitError("systolic array requires weight zero point = 0");
    auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto weightType = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!inputType || !weightType || !resultType || inputType.getRank() != 3 ||
        weightType.getRank() != 2 || resultType.getRank() != 3 ||
        !inputType.hasStaticShape() || !weightType.hasStaticShape() ||
        !resultType.hasStaticShape() ||
        !inputType.getElementType().isSignlessInteger(8) ||
        !weightType.getElementType().isSignlessInteger(8) ||
        !resultType.getElementType().isSignlessInteger(32))
      return op.emitError(
          "expected static signless i8 inputs and i32 accumulator");
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    int64_t batch = resultType.getDimSize(0);
    int64_t channels = resultType.getDimSize(1);
    int64_t positions = resultType.getDimSize(2);
    int64_t reduction = inputType.getDimSize(1);
    SmallVector<ReassociationIndices> reassoc{{0, 1}, {2}};
    Value lhs3 = transposeTensor(rewriter, loc, op.getInputs()[0], {0, 2, 1});
    Value lhs = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get({batch * positions, reduction},
                              rewriter.getI8Type()),
        lhs3, reassoc);
    Value rhs = transposeTensor(rewriter, loc, op.getInputs()[1], {1, 0});
    Value acc3 = transposeTensor(rewriter, loc, op.getOutputs()[0], {0, 2, 1});
    auto matrixType = RankedTensorType::get({batch * positions, channels},
                                            rewriter.getI32Type());
    Value acc = tensor::CollapseShapeOp::create(rewriter, loc, matrixType, acc3,
                                                reassoc);
    // Keep the scalar zero point captured in the region. Padding then has
    // exactly three shaped operands and can use the standard Linalg utility.
    auto maps = getU8S8MatmulIndexingMaps(&getContext());
    maps.erase(maps.begin() + 2);
    auto matmul = linalg::GenericOp::create(
        rewriter, loc, TypeRange{matrixType}, ValueRange{lhs, rhs},
        ValueRange{acc}, maps, getU8S8MatmulIteratorTypes(),
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value a = arith::ExtUIOp::create(b, l, b.getI32Type(), args[0]);
          Value w = arith::ExtSIOp::create(b, l, b.getI32Type(), args[1]);
          a = arith::SubIOp::create(b, l, a, op.getInputs()[2]);
          Value product = arith::MulIOp::create(b, l, a, w);
          Value sum = arith::AddIOp::create(b, l, args[2], product);
          linalg::YieldOp::create(b, l, sum);
        },
        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr(kU8S8MatmulMarker, rewriter.getUnitAttr())});
    linalg::LinalgPaddingOptions options;
    options.setPaddingDimensions({0, 1, 2})
        .setPadToMultipleOf({8, 8, 8})
        .setPaddingValues({rewriter.getI8IntegerAttr(inputZp.getSExtValue()),
                           rewriter.getI8IntegerAttr(0),
                           rewriter.getI32IntegerAttr(0)})
        .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);
    linalg::LinalgOp padded;
    SmallVector<Value> replacements;
    SmallVector<tensor::PadOp> pads;
    rewriter.setInsertionPoint(matmul);
    if (failed(linalg::rewriteAsPaddedOp(rewriter, matmul, options, padded,
                                         replacements, pads)))
      return op.emitError("failed to pad the quantized matrix multiplication");
    // The utility returns an unpadded slice and leaves replacing the op to us.
    Value matrix = replacements.front();
    rewriter.eraseOp(matmul);
    rewriter.setInsertionPoint(op);
    Value expanded = tensor::ExpandShapeOp::create(
        rewriter, loc,
        RankedTensorType::get({batch, positions, channels}, rewriter.getI32Type()),
        matrix, reassoc);
    Value result = transposeTensor(rewriter, loc, expanded, {0, 2, 1});
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Rank-2 epilogues have already been tiled together with their producers.
// Copy only the microtile at the runtime boundary, never a full accumulator.
// Entry-block allocas are reusable because runtime calls are synchronous.
static LogicalResult lowerMatrixEpilogue(Operation *op,
                                         PatternRewriter &rewriter) {
  auto outType = cast<RankedTensorType>(op->getResult(0).getType());
  if (outType.getShape() != ArrayRef<int64_t>({8, 8}))
    return rewriter.notifyMatchFailure(
        op, "expected an already tiled 8x8 matrix epilogue");
  auto function = op->getParentOfType<func::FuncOp>();
  if (!function)
    return failure();
  Location loc = op->getLoc();
  SmallVector<Value> buffers;
  for (Value operand : op->getOperands()) {
    auto type = cast<RankedTensorType>(operand.getType());
    auto element = cast<IntegerType>(type.getElementType());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&function.getBody().front());
    buffers.push_back(memref::AllocaOp::create(
        rewriter, loc,
        MemRefType::get(type.getShape(),
                        rewriter.getIntegerType(element.getWidth()))));
  }
  Value c0 = constantIndex(rewriter, loc, 0);
  Value c1 = constantIndex(rewriter, loc, 1);
  Value c8 = constantIndex(rewriter, loc, 8);
  for (unsigned i = 0; i + 1 < op->getNumOperands(); ++i) {
    Value input = op->getOperand(i), buffer = buffers[i];
    auto type = cast<RankedTensorType>(input.getType());
    auto store = [&](OpBuilder &b, Location l, ValueRange indices) {
      Value value = tensor::ExtractOp::create(b, l, input, indices);
      value = castIntegerSignedness(
          b, l, value, cast<MemRefType>(buffer.getType()).getElementType());
      memref::StoreOp::create(b, l, value, buffer, indices);
    };
    scf::ForOp::create(
        rewriter, loc, c0, c8, c1, ValueRange{},
        [&](OpBuilder &b, Location l, Value row, ValueRange) {
          if (type.getRank() == 1) {
            store(b, l, ValueRange{row});
          } else {
            scf::ForOp::create(
                b, l, c0, c8, c1, ValueRange{},
                [&](OpBuilder &nested, Location nl, Value col, ValueRange) {
                  store(nested, nl, ValueRange{row, col});
                  scf::YieldOp::create(nested, nl);
                });
          }
          scf::YieldOp::create(b, l);
        });
  }
  OperationState state(loc, op->getName());
  state.addOperands(buffers);
  state.addAttributes(op->getAttrs());
  rewriter.create(state);
  auto rows = scf::ForOp::create(
      rewriter, loc, c0, c8, c1, ValueRange{op->getOperands().back()},
      [&](OpBuilder &b, Location l, Value row, ValueRange args) {
        auto cols = scf::ForOp::create(
            b, l, c0, c8, c1, args,
            [&](OpBuilder &nested, Location nl, Value col,
                ValueRange innerArgs) {
              Value value = memref::LoadOp::create(nested, nl, buffers.back(),
                                                   ValueRange{row, col});
              value = castIntegerSignedness(nested, nl, value,
                                            outType.getElementType());
              Value result = tensor::InsertOp::create(
                  nested, nl, value, innerArgs.front(), ValueRange{row, col});
              scf::YieldOp::create(nested, nl, result);
            });
        scf::YieldOp::create(b, l, cols.getResult(0));
      });
  rewriter.replaceOp(op, rows.getResult(0));
  return success();
}

class LowerConvRequantToTilesPattern
    : public OpRewritePattern<standalone::ConvRequantOp> {
public:
  using OpRewritePattern<standalone::ConvRequantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(standalone::ConvRequantOp requant,
                                PatternRewriter &rewriter) const override {
    if (requant.getResult().empty())
      return failure();
    return lowerMatrixEpilogue(requant, rewriter);
  }
};

class LowerConvRequantToTiles
    : public impl::LowerConvRequantToTilesBase<LowerConvRequantToTiles> {
public:
  using impl::LowerConvRequantToTilesBase<
      LowerConvRequantToTiles>::LowerConvRequantToTilesBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerConvRequantToTilesPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    bool hasTensorForm = false;
    getOperation().walk([&](standalone::ConvRequantOp op) {
      if (!op.getResult().empty()) {
        op.emitError("expected a fused 8x8 matrix tile; run "
                     "fuse-matmul-epilogues before hardware buffer lowering");
        hasTensorForm = true;
      }
    });
    if (hasTensorForm)
      signalPassFailure();
  }
};

class LowerQAddReluToTilesPattern
    : public OpRewritePattern<standalone::QAddReluOp> {
public:
  using OpRewritePattern<standalone::QAddReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(standalone::QAddReluOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getResult().empty())
      return failure();
    return lowerMatrixEpilogue(op, rewriter);
  }
};

class LowerQAddReluToTiles
    : public impl::LowerQAddReluToTilesBase<LowerQAddReluToTiles> {
public:
  using impl::LowerQAddReluToTilesBase<
      LowerQAddReluToTiles>::LowerQAddReluToTilesBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerQAddReluToTilesPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    bool hasTensorForm = false;
    getOperation().walk([&](standalone::QAddReluOp op) {
      if (!op.getResult().empty()) {
        op.emitError("expected a fused 8x8 matrix tile; run "
                     "fuse-matmul-epilogues before hardware buffer lowering");
        hasTensorForm = true;
      }
    });
    if (hasTensorForm)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::standalone
