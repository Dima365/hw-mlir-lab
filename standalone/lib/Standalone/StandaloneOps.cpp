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

LogicalResult standalone::ConvRequantOp::verify() {
  auto accType = dyn_cast<MemRefType>(getAcc().getType());
  if (!accType || accType.getShape() != ArrayRef<int64_t>({8, 8}) ||
      !accType.getElementType().isInteger(32)) {
    return emitOpError("requires acc to be memref<8x8xi32>");
  }

  auto multiplierType = dyn_cast<MemRefType>(getMultiplier().getType());
  if (!multiplierType ||
      multiplierType.getShape() != ArrayRef<int64_t>({8}) ||
      !multiplierType.getElementType().isInteger(32)) {
    return emitOpError("requires multiplier to be memref<8xi32>");
  }

  auto shiftType = dyn_cast<MemRefType>(getShift().getType());
  if (!shiftType || shiftType.getShape() != ArrayRef<int64_t>({8}) ||
      !shiftType.getElementType().isInteger(32)) {
    return emitOpError("requires shift to be memref<8xi32>");
  }

  auto outType = dyn_cast<MemRefType>(getOut().getType());
  if (!outType || outType.getShape() != ArrayRef<int64_t>({8, 8}) ||
      !outType.getElementType().isInteger(8)) {
    return emitOpError("requires out to be memref<8x8xi8>");
  }

  int64_t outputZeroPoint = getOutputZeroPointAttr().getInt();
  if (outputZeroPoint < 0 || outputZeroPoint > 255)
    return emitOpError("requires output_zero_point in [0, 255]");

  int64_t reluEnable = getReluEnableAttr().getInt();
  if (reluEnable != 0 && reluEnable != 1)
    return emitOpError("requires relu_enable to be 0 or 1");

  return success();
}
