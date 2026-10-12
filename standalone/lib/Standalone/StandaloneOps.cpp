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
         memrefType.getDimSize(0) == 8 && memrefType.getDimSize(1) == 8 &&
         memrefType.getElementType().isInteger(width);
}

LogicalResult standalone::SystolicU8S8MatmulOp::verify() {
  auto lhsType = dyn_cast<MemRefType>(getLhs().getType());
  auto rhsType = dyn_cast<MemRefType>(getRhs().getType());
  auto accType = dyn_cast<MemRefType>(getAcc().getType());

  if (!lhsType || lhsType.getShape() != ArrayRef<int64_t>({8, 8}) ||
      !(lhsType.getElementType().isUnsignedInteger(8) ||
        lhsType.getElementType().isSignlessInteger(8)))
    return emitOpError("requires lhs to be memref<8x8xi8> or memref<8x8xui8>");
  if (!rhsType || rhsType.getShape() != ArrayRef<int64_t>({8, 8}) ||
      !rhsType.getElementType().isInteger(8) ||
      rhsType.getElementType().isUnsignedInteger())
    return emitOpError("requires rhs to be memref<8x8xi8>");
  if (!accType || accType.getShape() != ArrayRef<int64_t>({8, 8}) ||
      !accType.getElementType().isInteger(32))
    return emitOpError("requires acc to be memref<8x8xi32>");

  int64_t lhsZeroPoint = getLhsZeroPointAttr().getInt();
  if (lhsZeroPoint < 0 || lhsZeroPoint > 255)
    return emitOpError("requires lhs_zero_point in [0, 255]");

  return success();
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
    if ((lhsTensor.getRank() != 4 && lhsTensor.getRank() != 2) ||
        !lhsTensor.hasStaticShape() ||
        !lhsTensor.getElementType().isUnsignedInteger(8))
      return emitOpError("requires tensor lhs to be static rank-4 NCHW ui8 "
                         "or rank-2 [M, C] ui8");
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
    if ((accTensor.getRank() != 4 && accTensor.getRank() != 2) ||
        !accTensor.hasStaticShape() ||
        !accTensor.getElementType().isInteger(32))
      return emitOpError("requires tensor acc to be static rank-4 NCHW i32 "
                         "or rank-2 [M, C] i32");
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
