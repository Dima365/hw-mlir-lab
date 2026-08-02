//===- StandaloneTransformOps.h - Standalone transform ops -----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_STANDALONETRANSFORMOPS_H
#define STANDALONE_STANDALONETRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#define GET_OP_CLASSES
#include "Standalone/StandaloneTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace standalone {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace standalone
} // namespace mlir

#endif // STANDALONE_STANDALONETRANSFORMOPS_H
