//===- StandaloneTilingInterfaces.cpp - Epilogue tiling ---------------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Standalone/StandaloneOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;

namespace {
template <typename OpTy>
struct EpilogueTiling
    : TilingInterface::ExternalModel<EpilogueTiling<OpTy>, OpTy> {
  using Base = TilingInterface::ExternalModel<EpilogueTiling<OpTy>, OpTy>;
  using Base::generateResultTileValue;
  using Base::getTiledImplementation;

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto type = dyn_cast<RankedTensorType>(cast<OpTy>(op).getOut().getType());
    if (!type)
      return {};
    return SmallVector<utils::IteratorType>(type.getRank(),
                                            utils::IteratorType::parallel);
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto type = dyn_cast<RankedTensorType>(cast<OpTy>(op).getOut().getType());
    if (!type)
      return {};
    SmallVector<Range> ranges;
    for (int64_t size : type.getShape())
      ranges.push_back(
          {b.getIndexAttr(0), b.getIndexAttr(size), b.getIndexAttr(1)});
    return ranges;
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    if (op->getNumResults() != 1)
      return failure();
    auto type = cast<RankedTensorType>(op->getResult(0).getType());
    if (offsets.size() != size_t(type.getRank()) ||
        sizes.size() != offsets.size())
      return failure();
    // Hardware microtiles have fixed dimensions. Pad the logical tensors
    // before tiling, instead of constructing unverifiable dynamic tile types.
    if (llvm::any_of(sizes, [](OpFoldResult size) {
          return !getConstantIntValue(size).has_value();
        }))
      return failure();
    SmallVector<Value> operands;
    SmallVector<Operation *> slices;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      bool channelParameter =
          isa<standalone::ConvRequantOp>(op) && (index == 1 || index == 2);
      SmallVector<OpFoldResult> sliceOffsets(offsets), sliceSizes(sizes);
      if (channelParameter) {
        sliceOffsets = {offsets[1]};
        sliceSizes = {sizes[1]};
      }
      SmallVector<OpFoldResult> strides(sliceSizes.size(), b.getIndexAttr(1));
      auto slice = tensor::ExtractSliceOp::create(
          b, op->getLoc(), operand, sliceOffsets, sliceSizes, strides);
      operands.push_back(slice);
      slices.push_back(slice);
    }
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(operands.back().getType());
    state.addAttributes(op->getAttrs());
    Operation *tiled = b.create(state);
    return TilingResult{{tiled}, {tiled->getResult(0)}, slices};
  }

  LogicalResult
  getResultTilePosition(Operation *, OpBuilder &, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    if (resultNumber != 0)
      return failure();
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  LogicalResult getIterationDomainTileFromResultTile(
      Operation *, OpBuilder &, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterationOffsets,
      SmallVectorImpl<OpFoldResult> &iterationSizes) const {
    if (resultNumber != 0)
      return failure();
    iterationOffsets.assign(offsets.begin(), offsets.end());
    iterationSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    if (resultNumber != 0)
      return failure();
    return getTiledImplementation(op, b, offsets, sizes);
  }
};
} // namespace

void mlir::standalone::registerEpilogueTilingInterfaces(MLIRContext *context) {
  ConvRequantOp::attachInterface<EpilogueTiling<ConvRequantOp>>(*context);
  QAddReluOp::attachInterface<EpilogueTiling<QAddReluOp>>(*context);
}
