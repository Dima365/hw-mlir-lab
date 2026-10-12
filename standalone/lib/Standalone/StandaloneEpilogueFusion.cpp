//===- StandaloneEpilogueFusion.cpp - Shared accelerator output tiles
//------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/StandalonePasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::standalone {
#define GEN_PASS_DEF_FUSEMATMULEPILOGUES
#include "Standalone/StandalonePasses.h.inc"

namespace {
constexpr StringLiteral kMatmul = "standalone.u8s8_matmul";
constexpr StringLiteral kFuse = "standalone.fuse_epilogue";

static Value emptyLike(OpBuilder &b, Location loc, ArrayRef<int64_t> shape,
                       Type element) {
  return tensor::EmptyOp::create(b, loc, shape, element);
}

static Value transpose(OpBuilder &b, Location loc, Value value,
                       ArrayRef<int64_t> permutation) {
  auto type = cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> shape;
  for (int64_t dim : permutation)
    shape.push_back(type.getDimSize(dim));
  return linalg::TransposeOp::create(
             b, loc, value, emptyLike(b, loc, shape, type.getElementType()),
             permutation)
      .getResult()[0];
}

static Value padChannels(OpBuilder &b, Location loc, Value value,
                         int64_t channels) {
  auto type = cast<RankedTensorType>(value.getType());
  if (type.getDimSize(0) == channels)
    return value;
  Value zero = arith::ConstantIntOp::create(b, loc, 0, 32);
  return tensor::PadOp::create(
      b, loc, RankedTensorType::get({channels}, type.getElementType()), value,
      ArrayRef<OpFoldResult>{b.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(channels - type.getDimSize(0))},
      zero);
}

static Value restoreNchw(OpBuilder &b, Location loc, Value matrix,
                         RankedTensorType type) {
  int64_t batch = type.getDimSize(0), channels = type.getDimSize(1);
  int64_t positions = type.getDimSize(2) * type.getDimSize(3);
  Value unpadded = tensor::ExtractSliceOp::create(
      b, loc, matrix,
      ArrayRef<OpFoldResult>{b.getIndexAttr(0), b.getIndexAttr(0)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(batch * positions),
                             b.getIndexAttr(channels)},
      ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
  Value expanded = tensor::ExpandShapeOp::create(
      b, loc,
      RankedTensorType::get({batch, positions, channels},
                            type.getElementType()),
      unpadded, ArrayRef<ReassociationIndices>{{0, 1}, {2}});
  Value nchw = transpose(b, loc, expanded, {0, 2, 1});
  return tensor::ExpandShapeOp::create(
      b, loc, type, nchw, ArrayRef<ReassociationIndices>{{0}, {1}, {2, 3}});
}

// Accept only the layout bridge emitted by PrepareU8S8Matmul. Checking each
// reassociation/permutation avoids confusing NCHW order with position-major M.
static linalg::GenericOp findMatmul(Value acc) {
  auto nchw = acc.getDefiningOp<tensor::ExpandShapeOp>();
  if (!nchw || !acc.hasOneUse() ||
      nchw.getReassociationIndices() !=
          SmallVector<ReassociationIndices>{{0}, {1}, {2, 3}})
    return {};
  auto trans = nchw.getSrc().getDefiningOp<linalg::TransposeOp>();
  if (!trans || !trans->hasOneUse() ||
      trans.getPermutation() != ArrayRef<int64_t>({0, 2, 1}))
    return {};
  auto expand = trans.getInput().getDefiningOp<tensor::ExpandShapeOp>();
  if (!expand || !expand->hasOneUse() ||
      expand.getReassociationIndices() !=
          SmallVector<ReassociationIndices>{{0, 1}, {2}})
    return {};
  Value matrix = expand.getSrc();
  if (auto slice = matrix.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (!slice->hasOneUse() ||
        slice.getStaticOffsets() != ArrayRef<int64_t>({0, 0}) ||
        slice.getStaticStrides() != ArrayRef<int64_t>({1, 1}) ||
        slice.getSourceType().getRank() != 2)
      return {};
    matrix = slice.getSource();
  }
  auto matmul = matrix.getDefiningOp<linalg::GenericOp>();
  if (!matmul || !matrix.hasOneUse() || !matmul->hasAttr(kMatmul) ||
      matmul.getInputs().size() != 2 || matmul.getOutputs().size() != 1 ||
      !matmul.hasPureTensorSemantics())
    return {};
  auto type = cast<RankedTensorType>(matrix.getType());
  auto original = cast<RankedTensorType>(acc.getType());
  int64_t m =
      original.getDimSize(0) * original.getDimSize(2) * original.getDimSize(3);
  int64_t c = original.getDimSize(1);
  if (m <= 0 || c <= 0)
    return {};
  if (type.getShape() !=
      ArrayRef<int64_t>({((m + 7) / 8) * 8, ((c + 7) / 8) * 8}))
    return {};
  return matmul;
}

static Value findBias(linalg::GenericOp matmul, RankedTensorType accType) {
  Value init = matmul.getOutputs()[0];
  if (auto pad = init.getDefiningOp<tensor::PadOp>()) {
    APInt padding;
    Value value = pad.getConstantPaddingValue();
    if (!value || !matchPattern(value, m_ConstantInt(&padding)) ||
        !padding.isZero() || pad.getStaticLow() != ArrayRef<int64_t>({0, 0}))
      return {};
    init = pad.getSource();
  }
  auto matrix = init.getDefiningOp<tensor::CollapseShapeOp>();
  if (!matrix || matrix.getReassociationIndices() !=
                     SmallVector<ReassociationIndices>{{0, 1}, {2}})
    return {};
  auto trans = matrix.getSrc().getDefiningOp<linalg::TransposeOp>();
  if (!trans || trans.getPermutation() != ArrayRef<int64_t>({0, 2, 1}))
    return {};
  auto collapsed = trans.getInput().getDefiningOp<tensor::CollapseShapeOp>();
  if (!collapsed || collapsed.getReassociationIndices() !=
                        SmallVector<ReassociationIndices>{{0}, {1}, {2, 3}})
    return {};
  auto broadcast = collapsed.getSrc().getDefiningOp<linalg::BroadcastOp>();
  if (!broadcast || broadcast.getDimensions() != ArrayRef<int64_t>({0, 2, 3}) ||
      broadcast.getResult()[0].getType() != accType)
    return {};
  return broadcast.getInput();
}

// A tileable gather, rather than a full transpose of the residual tensor.
// linalg.index is offset by standard tiling, including tiles crossing batches.
static Value gatherResidual(OpBuilder &b, Location loc, Value source,
                            ArrayRef<int64_t> matrixShape, int64_t zp) {
  auto type = cast<RankedTensorType>(source.getType());
  int64_t height = type.getDimSize(2), width = type.getDimSize(3);
  Value out = emptyLike(b, loc, matrixShape, type.getElementType());
  auto gather = linalg::GenericOp::create(
      b, loc, TypeRange{out.getType()}, ValueRange{}, ValueRange{out},
      ArrayRef<AffineMap>{b.getMultiDimIdentityMap(2)},
      ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                    utils::IteratorType::parallel},
      [&](OpBuilder &nested, Location l, ValueRange) {
        auto idx = [&](int64_t value) -> Value {
          return arith::ConstantIndexOp::create(nested, l, value);
        };
        Value m = linalg::IndexOp::create(nested, l, 0);
        Value c = linalg::IndexOp::create(nested, l, 1);
        Value validM =
            arith::CmpIOp::create(nested, l, arith::CmpIPredicate::ult, m,
                                  idx(type.getDimSize(0) * height * width));
        Value validC = arith::CmpIOp::create(
            nested, l, arith::CmpIPredicate::ult, c, idx(type.getDimSize(1)));
        Value valid = arith::AndIOp::create(nested, l, validM, validC);
        auto select = scf::IfOp::create(
            nested, l, valid,
            [&](OpBuilder &then, Location tl) {
              Value hw =
                  arith::ConstantIndexOp::create(then, tl, height * width);
              Value w = arith::ConstantIndexOp::create(then, tl, width);
              Value n = arith::DivUIOp::create(then, tl, m, hw);
              Value p = arith::RemUIOp::create(then, tl, m, hw);
              Value y = arith::DivUIOp::create(then, tl, p, w);
              Value x = arith::RemUIOp::create(then, tl, p, w);
              Value element = tensor::ExtractOp::create(then, tl, source,
                                                        ValueRange{n, c, y, x});
              scf::YieldOp::create(then, tl, element);
            },
            [&](OpBuilder &otherwise, Location el) {
              Value bits = arith::ConstantIntOp::create(otherwise, el, zp, 8);
              Value zero =
                  UnrealizedConversionCastOp::create(
                      otherwise, el, TypeRange{type.getElementType()}, bits)
                      .getResult(0);
              scf::YieldOp::create(otherwise, el, zero);
            });
        linalg::YieldOp::create(nested, l, select.getResult(0));
      });
  gather->setAttr(kFuse, b.getUnitAttr());
  return gather.getResult(0);
}

class FuseMatmulEpilogues
    : public impl::FuseMatmulEpiloguesBase<FuseMatmulEpilogues> {
public:
  using impl::FuseMatmulEpiloguesBase<
      FuseMatmulEpilogues>::FuseMatmulEpiloguesBase;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    DenseMap<Value, Value> matrices;
    SmallVector<ConvRequantOp> requants;
    getOperation().walk([&](ConvRequantOp op) {
      if (auto type = dyn_cast<RankedTensorType>(op.getAcc().getType());
          type && type.getRank() == 4)
        requants.push_back(op);
    });
    for (auto requant : requants) {
      auto matmul = findMatmul(requant.getAcc());
      if (!matmul) {
        requant.emitError(
            "fusion requires a single-use prepared matmul accumulator "
            "with the expected NCHW layout bridge");
        signalPassFailure();
        return;
      }
      auto accType = cast<RankedTensorType>(requant.getAcc().getType());
      Value bias = findBias(matmul, accType);
      if (!bias) {
        requant.emitError("fusion requires channel-broadcast bias initialization");
        signalPassFailure();
        return;
      }
      // Boundary activations may feed both branches of the next block. Only
      // a requantized value consumed by QADD must be private to that QADD.
      if (!requant->hasOneUse() &&
          llvm::any_of(requant->getResult(0).getUsers(), [](Operation *user) {
            return isa<QAddReluOp>(user);
          })) {
        requant.emitError(
            "requantized input to QADD must have a single use; "
            "separate QADD scheduling is not supported");
        signalPassFailure();
        return;
      }
      auto matrixType = cast<RankedTensorType>(matmul.getResult(0).getType());
      auto shape = matrixType.getShape();
      rewriter.setInsertionPoint(matmul);
      Location loc = requant.getLoc();
      Value paddedBias = padChannels(rewriter, loc, bias, shape[1]);
      auto initial = linalg::BroadcastOp::create(
          rewriter, loc, paddedBias,
          emptyLike(rewriter, loc, shape, rewriter.getI32Type()),
          ArrayRef<int64_t>{0});
      initial->setAttr(kFuse, rewriter.getUnitAttr());
      matmul.getOutputsMutable().assign(initial.getResult());
      matmul->setAttr(kFuse, rewriter.getUnitAttr());
      rewriter.setInsertionPoint(requant);
      Value multiplier =
          padChannels(rewriter, loc, requant.getMultiplier(), shape[1]);
      Value shift = padChannels(rewriter, loc, requant.getShift(), shape[1]);
      auto outputType = cast<RankedTensorType>(requant.getOut().getType());
      Value out = emptyLike(rewriter, loc, shape, outputType.getElementType());
      auto normalized = ConvRequantOp::create(
          rewriter, loc, TypeRange{out.getType()}, matmul.getResult(0),
          multiplier, shift, out, requant.getOutputZeroPointAttr(),
          requant.getReluEnableAttr());
      normalized->setAttr(kFuse, rewriter.getUnitAttr());
      Value restored =
          restoreNchw(rewriter, loc, normalized.getResult()[0], outputType);
      if (requant->hasOneUse())
        matrices[restored] = normalized.getResult()[0];
      rewriter.replaceOp(requant, restored);
    }

    SmallVector<QAddReluOp> adds;
    getOperation().walk([&](QAddReluOp op) {
      if (auto type = dyn_cast<RankedTensorType>(op.getLhs().getType());
          type && type.getRank() == 4)
        adds.push_back(op);
    });
    for (auto add : adds) {
      Value lhs = matrices.lookup(add.getLhs());
      Value rhs = matrices.lookup(add.getRhs());
      if (!lhs && !rhs) {
        add.emitError("QADD fusion requires at least one single-use "
                      "matmul -> conv_requant producer");
        signalPassFailure();
        return;
      }
      rewriter.setInsertionPoint(add);
      Location loc = add.getLoc();
      auto matrixType = cast<RankedTensorType>((lhs ? lhs : rhs).getType());
      if (!lhs)
        lhs = gatherResidual(rewriter, loc, add.getLhs(), matrixType.getShape(),
                             add.getLhsZeroPointAttr().getInt());
      if (!rhs)
        rhs = gatherResidual(rewriter, loc, add.getRhs(), matrixType.getShape(),
                             add.getRhsZeroPointAttr().getInt());
      Value out = emptyLike(rewriter, loc, matrixType.getShape(),
                            matrixType.getElementType());
      auto normalized = QAddReluOp::create(
          rewriter, loc, TypeRange{out.getType()}, lhs, rhs, out,
          add.getLhsMultiplierAttr(), add.getRhsMultiplierAttr(),
          add.getShiftAttr(), add.getLhsZeroPointAttr(),
          add.getRhsZeroPointAttr(), add.getOutputZeroPointAttr(),
          add.getReluEnableAttr());
      normalized->setAttr(kFuse, rewriter.getUnitAttr());
      rewriter.replaceOp(
          add, restoreNchw(rewriter, loc, normalized.getResult()[0],
                           cast<RankedTensorType>(add.getOut().getType())));
    }

    // Remove obsolete NCHW bridges so fused producers can become dead.
    if (failed(cleanup())) {
      signalPassFailure();
      return;
    }
    for (;;) {
      Operation *root = nullptr;
      getOperation().walk([&](QAddReluOp op) {
        if (op->hasAttr(kFuse))
          root = op;
      });
      if (!root)
        getOperation().walk([&](ConvRequantOp op) {
          if (op->hasAttr(kFuse))
            root = op;
        });
      if (!root)
        break;
      rewriter.setInsertionPoint(root);
      scf::SCFTileAndFuseOptions options;
      options.tilingOptions.setTileSizes(
          {rewriter.getIndexAttr(8), rewriter.getIndexAttr(8)});
      options.setFusionControlFn(
          [](tensor::ExtractSliceOp, OpResult producer, bool)
              -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
            Operation *op = producer.getOwner();
            if (op->hasAttr(kFuse))
              return scf::SCFTileAndFuseOptions::ControlFnResult{};
            return std::nullopt;
          });
      auto fused = scf::tileConsumerAndFuseProducersUsingSCF(
          rewriter, cast<TilingInterface>(root), options);
      if (failed(fused)) {
        root->emitError("failed to fuse accelerator epilogue output tiles");
        signalPassFailure();
        return;
      }
      for (Operation *tiled : fused->tiledAndFusedOps) {
        // Standard fusion slices the original full-size empty destinations.
        // These operations overwrite every element; give each a reusable
        // microtile destination so bufferization cannot retain the full tensor.
        if (isa<ConvRequantOp, QAddReluOp, linalg::BroadcastOp>(tiled) ||
            (isa<linalg::GenericOp>(tiled) && tiled->hasAttr(kFuse) &&
             !tiled->hasAttr(kMatmul))) {
          auto dps = cast<DestinationStyleOpInterface>(tiled);
          auto type = cast<RankedTensorType>(dps.getDpsInits()[0].getType());
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(fused->loops.front());
          dps.setDpsInitOperand(0, emptyLike(rewriter, tiled->getLoc(),
                                             type.getShape(),
                                             type.getElementType()));
        }
        tiled->removeAttr(kFuse);
      }
      rewriter.replaceOp(root, fused->replacements.lookup(root->getResult(0)));
      if (failed(cleanup())) {
        signalPassFailure();
        return;
      }
    }
    bool invalidEpilogue = false;
    getOperation().walk([&](Operation *op) {
      op->removeAttr(kFuse);
      if (!isa<ConvRequantOp, QAddReluOp>(op) || op->getNumResults() == 0)
        return;
      auto type = cast<RankedTensorType>(op->getResult(0).getType());
      if (type.getShape() != ArrayRef<int64_t>({8, 8})) {
        op->emitError("fusion must produce an 8x8 matrix epilogue tile");
        invalidEpilogue = true;
      }
    });
    if (invalidEpilogue)
      signalPassFailure();
  }

private:
  LogicalResult cleanup() {
    return applyPatternsGreedily(getOperation(),
                                 RewritePatternSet(&getContext()));
  }
};
} // namespace
} // namespace mlir::standalone
