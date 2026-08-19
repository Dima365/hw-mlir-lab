//===- StandaloneOps.cpp - Standalone dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneOps.h"
#include "Standalone/StandaloneDialect.h"

#define GET_OP_CLASSES
#include "Standalone/StandaloneOps.cpp.inc"

using namespace mlir;

static bool is8x8IntegerMemRef(Type type, unsigned width) {
  auto memrefType = dyn_cast<MemRefType>(type);
  return memrefType && memrefType.getRank() == 2 &&
         memrefType.getDimSize(0) == 8 &&
         memrefType.getDimSize(1) == 8 &&
         memrefType.getElementType().isInteger(width);
}

LogicalResult standalone::QAddReluOp::verify() {
  auto lhsMemRef = dyn_cast<MemRefType>(getLhs().getType());
  auto rhsMemRef = dyn_cast<MemRefType>(getRhs().getType());
  auto outMemRef = dyn_cast<MemRefType>(getOut().getType());
  bool hasAnyMemRef = lhsMemRef || rhsMemRef || outMemRef;

  if (hasAnyMemRef) {
    if (!lhsMemRef || !rhsMemRef || !outMemRef)
      return emitOpError("does not allow mixed tensor and memref operands");
    if (!getResult().empty())
      return emitOpError("buffer form must not have an SSA result");
    if (!is8x8IntegerMemRef(getLhs().getType(), 8))
      return emitOpError("requires buffer lhs to be memref<8x8xi8>");
    if (!is8x8IntegerMemRef(getRhs().getType(), 8))
      return emitOpError("requires buffer rhs to be memref<8x8xi8>");
    if (!is8x8IntegerMemRef(getOut().getType(), 8))
      return emitOpError("requires buffer out to be memref<8x8xi8>");
  } else {
    auto lhsTensor = dyn_cast<RankedTensorType>(getLhs().getType());
    auto rhsTensor = dyn_cast<RankedTensorType>(getRhs().getType());
    auto outTensor = dyn_cast<RankedTensorType>(getOut().getType());
    if (!lhsTensor || !rhsTensor || !outTensor)
      return emitOpError("requires all operands to be ranked tensors");
    if (getResult().size() != 1 || getResult().front().getType() != outTensor)
      return emitOpError(
          "tensor form requires one result with the output tensor type");
    if (lhsTensor.getRank() != 4 || !lhsTensor.hasStaticShape() ||
        !lhsTensor.getElementType().isUnsignedInteger(8))
      return emitOpError("requires tensor lhs to be static rank-4 NCHW ui8");
    if (rhsTensor != lhsTensor || outTensor != lhsTensor)
      return emitOpError(
          "requires tensor rhs and out to have the lhs tensor type");
  }

  if (getLhsMultiplierAttr().getInt() < 0 ||
      getRhsMultiplierAttr().getInt() < 0) {
    return emitOpError("requires non-negative multipliers");
  }

  int64_t shift = getShiftAttr().getInt();
  if (shift < 0 || shift > 63)
    return emitOpError("requires shift in [0, 63]");

  int64_t lhsZeroPoint = getLhsZeroPointAttr().getInt();
  if (lhsZeroPoint < 0 || lhsZeroPoint > 255)
    return emitOpError("requires lhs_zero_point in [0, 255]");

  int64_t rhsZeroPoint = getRhsZeroPointAttr().getInt();
  if (rhsZeroPoint < 0 || rhsZeroPoint > 255)
    return emitOpError("requires rhs_zero_point in [0, 255]");

  int64_t outputZeroPoint = getOutputZeroPointAttr().getInt();
  if (outputZeroPoint < 0 || outputZeroPoint > 255)
    return emitOpError("requires output_zero_point in [0, 255]");

  int64_t reluEnable = getReluEnableAttr().getInt();
  if (reluEnable != 0 && reluEnable != 1)
    return emitOpError("requires relu_enable to be 0 or 1");

  return success();
}

void standalone::QAddReluOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!isa<MemRefType>(getLhs().getType()))
    return;

  effects.emplace_back(MemoryEffects::Read::get(),
                       &getOperation()->getOpOperand(0), 0, false,
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       &getOperation()->getOpOperand(1), 0, false,
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       &getOperation()->getOpOperand(2), 0, false,
                       SideEffects::DefaultResource::get());
}

LogicalResult standalone::ConvAccumulateOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto resultType = dyn_cast<RankedTensorType>(getResult().getType());

  if (!inputType || inputType.getRank() != 4 || !inputType.hasStaticShape() ||
      !inputType.getElementType().isUnsignedInteger(8))
    return emitOpError("requires input to be a static rank-4 NCHW ui8 tensor");
  if (!weightType || weightType.getRank() != 4 ||
      !weightType.hasStaticShape() ||
      !(weightType.getElementType().isSignlessInteger(8) ||
        weightType.getElementType().isSignedInteger(8)))
    return emitOpError("requires weight to be a static rank-4 OIHW i8 tensor");
  if (!biasType || biasType.getRank() != 1 || !biasType.hasStaticShape() ||
      !(biasType.getElementType().isSignlessInteger(32) ||
        biasType.getElementType().isSignedInteger(32)))
    return emitOpError("requires bias to be a static rank-1 i32 tensor");
  if (!resultType || resultType.getRank() != 4 ||
      !resultType.hasStaticShape() ||
      !(resultType.getElementType().isSignlessInteger(32) ||
        resultType.getElementType().isSignedInteger(32)))
    return emitOpError("requires result to be a static rank-4 NCHW i32 tensor");

  int64_t inputZeroPoint = getInputZeroPointAttr().getInt();
  if (inputZeroPoint < 0 || inputZeroPoint > 255)
    return emitOpError("requires input_zero_point in [0, 255]");
  if (getGroupAttr().getInt() != 1)
    return emitOpError("currently requires group = 1");

  ArrayRef<int64_t> strides = getStrides();
  ArrayRef<int64_t> pads = getPads();
  ArrayRef<int64_t> dilations = getDilations();
  if (strides.size() != 2 || strides[0] <= 0 || strides[1] <= 0)
    return emitOpError("requires two positive strides");
  if (dilations.size() != 2 || dilations[0] <= 0 || dilations[1] <= 0)
    return emitOpError("requires two positive dilations");
  if (pads.size() != 4 ||
      llvm::any_of(pads, [](int64_t value) { return value < 0; }))
    return emitOpError("requires four non-negative pads");

  int64_t batch = inputType.getDimSize(0);
  int64_t inputChannels = inputType.getDimSize(1);
  int64_t inputHeight = inputType.getDimSize(2);
  int64_t inputWidth = inputType.getDimSize(3);
  int64_t outputChannels = weightType.getDimSize(0);
  int64_t kernelHeight = weightType.getDimSize(2);
  int64_t kernelWidth = weightType.getDimSize(3);

  if (weightType.getDimSize(1) != inputChannels)
    return emitOpError("requires weight input channels to match input C");
  if (biasType.getDimSize(0) != outputChannels)
    return emitOpError("requires bias length to match weight Cout");

  int64_t effectiveKernelHeight = dilations[0] * (kernelHeight - 1) + 1;
  int64_t effectiveKernelWidth = dilations[1] * (kernelWidth - 1) + 1;
  int64_t heightNumerator =
      inputHeight + pads[0] + pads[2] - effectiveKernelHeight;
  int64_t widthNumerator =
      inputWidth + pads[1] + pads[3] - effectiveKernelWidth;
  if (heightNumerator < 0 || widthNumerator < 0)
    return emitOpError("has a kernel larger than the padded input");

  SmallVector<int64_t> expectedResultShape = {batch, outputChannels,
                                              heightNumerator / strides[0] + 1,
                                              widthNumerator / strides[1] + 1};
  if (resultType.getShape() != ArrayRef<int64_t>(expectedResultShape))
    return emitOpError("result shape does not match convolution geometry");

  return success();
}

void standalone::ConvRequantOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (!isa<MemRefType>(getAcc().getType()))
    return;

  effects.emplace_back(MemoryEffects::Read::get(),
                       &getOperation()->getOpOperand(0), 0, false,
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       &getOperation()->getOpOperand(1), 0, false,
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       &getOperation()->getOpOperand(2), 0, false,
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       &getOperation()->getOpOperand(3), 0, false,
                       SideEffects::DefaultResource::get());
}

LogicalResult standalone::ConvRequantOp::verify() {
  auto accMemRef = dyn_cast<MemRefType>(getAcc().getType());
  auto multiplierMemRef = dyn_cast<MemRefType>(getMultiplier().getType());
  auto shiftMemRef = dyn_cast<MemRefType>(getShift().getType());
  auto outMemRef = dyn_cast<MemRefType>(getOut().getType());
  bool hasAnyMemRef = accMemRef || multiplierMemRef || shiftMemRef || outMemRef;

  if (hasAnyMemRef) {
    if (!accMemRef || !multiplierMemRef || !shiftMemRef || !outMemRef)
      return emitOpError("does not allow mixed tensor and memref operands");
    if (!getResult().empty())
      return emitOpError("buffer form must not have an SSA result");
    if (accMemRef.getShape() != ArrayRef<int64_t>({8, 8}) ||
        !accMemRef.getElementType().isInteger(32))
      return emitOpError("requires buffer acc to be memref<8x8xi32>");
    if (multiplierMemRef.getShape() != ArrayRef<int64_t>({8}) ||
        !multiplierMemRef.getElementType().isInteger(32))
      return emitOpError("requires buffer multiplier to be memref<8xi32>");
    if (shiftMemRef.getShape() != ArrayRef<int64_t>({8}) ||
        !shiftMemRef.getElementType().isInteger(32))
      return emitOpError("requires buffer shift to be memref<8xi32>");
    if (outMemRef.getShape() != ArrayRef<int64_t>({8, 8}) ||
        !outMemRef.getElementType().isInteger(8))
      return emitOpError("requires buffer out to be memref<8x8xi8>");
  } else {
    auto accTensor = dyn_cast<RankedTensorType>(getAcc().getType());
    auto multiplierTensor =
        dyn_cast<RankedTensorType>(getMultiplier().getType());
    auto shiftTensor = dyn_cast<RankedTensorType>(getShift().getType());
    auto outTensor = dyn_cast<RankedTensorType>(getOut().getType());
    if (!accTensor || !multiplierTensor || !shiftTensor || !outTensor)
      return emitOpError("requires all operands to be ranked tensors");
    if (getResult().size() != 1 || getResult().front().getType() != outTensor)
      return emitOpError(
          "tensor form requires one result with the output tensor type");
    if (accTensor.getRank() != 4 || !accTensor.hasStaticShape() ||
        !accTensor.getElementType().isInteger(32))
      return emitOpError("requires tensor acc to be static rank-4 NCHW i32");
    if (outTensor.getShape() != accTensor.getShape() ||
        !outTensor.getElementType().isUnsignedInteger(8))
      return emitOpError(
          "requires tensor out to match acc shape with ui8 elements");

    int64_t outputChannels = accTensor.getDimSize(1);
    if (multiplierTensor.getShape() != ArrayRef<int64_t>({outputChannels}) ||
        !multiplierTensor.getElementType().isInteger(32))
      return emitOpError(
          "requires one i32 tensor multiplier per output channel");
    if (shiftTensor.getShape() != ArrayRef<int64_t>({outputChannels}) ||
        !shiftTensor.getElementType().isInteger(32))
      return emitOpError("requires one i32 tensor shift per output channel");
  }

  int64_t outputZeroPoint = getOutputZeroPointAttr().getInt();
  if (outputZeroPoint < 0 || outputZeroPoint > 255)
    return emitOpError("requires output_zero_point in [0, 255]");

  int64_t reluEnable = getReluEnableAttr().getInt();
  if (reluEnable != 0 && reluEnable != 1)
    return emitOpError("requires relu_enable to be 0 or 1");

  return success();
}
