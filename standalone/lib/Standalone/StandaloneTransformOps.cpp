//===- StandaloneTransformOps.cpp - Standalone transform ops -------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneTransformOps.h"

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::transform;

namespace {

struct ConvQuantParams {
  double activationScale;
  int64_t activationZeroPoint;
  SmallVector<double> weightScales;
  SmallVector<int64_t> weightZeroPoints;
  double outputScale;
  int64_t outputZeroPoint;
  SmallVector<double> biasScales;
  SmallVector<int64_t> biasZeroPoints;
};

struct ConvRequantFixedPointParams {
  SmallVector<int64_t> multipliers;
  SmallVector<int64_t> shifts;
};

struct ConvAccumulateSpec {
  Value input;
  Value weight;
  Value bias;
  RankedTensorType resultType;
  int64_t inputZeroPoint;
  SmallVector<int64_t> strides;
  SmallVector<int64_t> pads;
  SmallVector<int64_t> dilations;
  int64_t group;
};

struct ConvRequantParams {
  DenseI64ArrayAttr multipliers;
  DenseI64ArrayAttr shifts;
  int64_t outputZeroPoint;
};

struct ConvRequantRewrite {
  Operation *conv;
  Operation *relu;
  Operation *quantize;
  ConvAccumulateSpec accumulate;
  DenseI64ArrayAttr multipliers;
  DenseI64ArrayAttr shifts;
  RankedTensorType outputType;
  int64_t outputZeroPoint;
};

struct QAddChain {
  Operation *lhsDQ;
  Operation *rhsDQ;
  Operation *add;
  Operation *relu;
  Operation *quantize;
};

struct QAddQuantParams {
  double lhsScale;
  int64_t lhsZeroPoint;
  double rhsScale;
  int64_t rhsZeroPoint;
  double outputScale;
  int64_t outputZeroPoint;
};

struct QAddFixedPointParams {
  int64_t lhsMultiplier;
  int64_t rhsMultiplier;
  int64_t shift;
};

struct QAddRewrite {
  QAddChain chain;
  RankedTensorType outputType;
  QAddFixedPointParams fixedPoint;
  int64_t lhsZeroPoint;
  int64_t rhsZeroPoint;
  int64_t outputZeroPoint;
};

static LogicalResult fail(std::string &error, StringRef message) {
  error = message.str();
  return failure();
}

static bool hasName(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}

static LogicalResult getDenseConstant(Value value, StringRef label,
                                      DenseElementsAttr &result,
                                      std::string &error) {
  Operation *constant = value.getDefiningOp();
  if (!hasName(constant, "onnx.Constant"))
    return fail(error, (label + " must be produced by onnx.Constant").str());

  result = constant->getAttrOfType<DenseElementsAttr>("value");
  if (!result)
    return fail(error,
                (label + " onnx.Constant must contain dense 'value'").str());
  return success();
}

static LogicalResult getDQ(Value value, StringRef label, Operation *&dq,
                           std::string &error) {
  dq = value.getDefiningOp();
  if (!hasName(dq, "onnx.DequantizeLinear"))
    return fail(error,
                (label + " must be produced by onnx.DequantizeLinear").str());
  if (dq->getNumOperands() != 3)
    return fail(error,
                (label + " DequantizeLinear must have three operands").str());
  return success();
}

static bool isRankedTensorOf(DenseElementsAttr attr, int64_t rank,
                             function_ref<bool(Type)> elementCheck) {
  auto type = dyn_cast<RankedTensorType>(attr.getType());
  return type && type.getRank() == rank && elementCheck(type.getElementType());
}

static bool isF32(Type type) { return type.isF32(); }
static bool isUI8(Type type) { return type.isUnsignedInteger(8); }
static bool isI8(Type type) {
  return type.isSignlessInteger(8) || type.isSignedInteger(8);
}
static bool isI32(Type type) {
  return type.isSignlessInteger(32) || type.isSignedInteger(32);
}

static LogicalResult readScalarFloat(DenseElementsAttr attr, StringRef label,
                                     double &result, std::string &error) {
  if (!isRankedTensorOf(attr, 0, isF32))
    return fail(error, (label + " must be tensor<f32>").str());
  result = (*attr.getValues<APFloat>().begin()).convertToDouble();
  if (!std::isfinite(result) || result <= 0.0)
    return fail(error, (label + " must be positive and finite").str());
  return success();
}

static LogicalResult readScalarUI8(DenseElementsAttr attr, StringRef label,
                                   int64_t &result, std::string &error) {
  if (!isRankedTensorOf(attr, 0, isUI8))
    return fail(error, (label + " must be tensor<ui8>").str());
  result = (*attr.getValues<APInt>().begin()).getZExtValue();
  return success();
}

static LogicalResult readFloatVector(DenseElementsAttr attr, StringRef label,
                                     int64_t expectedSize,
                                     SmallVectorImpl<double> &result,
                                     std::string &error) {
  if (!isRankedTensorOf(attr, 1, isF32) ||
      attr.getNumElements() != expectedSize)
    return fail(error, (label + " must be tensor<Coutxf32>").str());

  for (APFloat value : attr.getValues<APFloat>()) {
    double converted = value.convertToDouble();
    if (!std::isfinite(converted) || converted <= 0.0)
      return fail(error, (label + " values must be positive and finite").str());
    result.push_back(converted);
  }
  return success();
}

static LogicalResult readIntVector(DenseElementsAttr attr, StringRef label,
                                   int64_t expectedSize,
                                   function_ref<bool(Type)> elementCheck,
                                   SmallVectorImpl<int64_t> &result,
                                   std::string &error) {
  if (!isRankedTensorOf(attr, 1, elementCheck) ||
      attr.getNumElements() != expectedSize)
    return fail(error, (label + " has an unsupported type or shape").str());

  for (APInt value : attr.getValues<APInt>())
    result.push_back(value.getSExtValue());
  return success();
}

static LogicalResult requireTensor(Value value, int64_t rank,
                                   function_ref<bool(Type)> elementCheck,
                                   StringRef label, std::string &error) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != rank || !elementCheck(type.getElementType()))
    return fail(error, (label + " has an unsupported tensor type").str());
  return success();
}

static LogicalResult requireAxisZero(Operation *op, StringRef label,
                                     std::string &error) {
  auto axis = op->getAttrOfType<IntegerAttr>("axis");
  if (!axis || axis.getValue().getSExtValue() != 0)
    return fail(error, (label + " requires axis = 0").str());
  return success();
}

static LogicalResult extractConvQuantParams(Operation *conv,
                                            ConvQuantParams &params,
                                            std::string &error) {
  if (!hasName(conv, "onnx.Conv"))
    return fail(error, "target must be onnx.Conv");
  if (conv->getNumOperands() != 3 || conv->getNumResults() != 1)
    return fail(error,
                "onnx.Conv must have activation, weight and bias operands");

  auto outputType = dyn_cast<RankedTensorType>(conv->getResult(0).getType());
  if (!outputType || outputType.getRank() != 4 || outputType.isDynamicDim(1))
    return fail(error, "onnx.Conv result must be static rank-4 NCHW tensor");
  int64_t outputChannels = outputType.getDimSize(1);

  Operation *activationDQ;
  Operation *weightDQ;
  Operation *biasDQ;
  if (failed(getDQ(conv->getOperand(0), "activation", activationDQ, error)) ||
      failed(getDQ(conv->getOperand(1), "weight", weightDQ, error)) ||
      failed(getDQ(conv->getOperand(2), "bias", biasDQ, error)))
    return failure();

  if (failed(requireTensor(activationDQ->getOperand(0), 4, isUI8,
                           "quantized activation", error)) ||
      failed(requireTensor(weightDQ->getOperand(0), 4, isI8, "quantized weight",
                           error)) ||
      failed(requireTensor(biasDQ->getOperand(0), 1, isI32, "quantized bias",
                           error)) ||
      failed(requireAxisZero(weightDQ, "weight DequantizeLinear", error)) ||
      failed(requireAxisZero(biasDQ, "bias DequantizeLinear", error)))
    return failure();

  auto weightType = cast<RankedTensorType>(weightDQ->getOperand(0).getType());
  auto biasType = cast<RankedTensorType>(biasDQ->getOperand(0).getType());
  if (weightType.isDynamicDim(0) || weightType.getDimSize(0) != outputChannels)
    return fail(error, "quantized weight dimension 0 must equal Cout");
  if (biasType.isDynamicDim(0) || biasType.getDimSize(0) != outputChannels)
    return fail(error, "quantized bias length must equal Cout");

  DenseElementsAttr activationScaleAttr;
  DenseElementsAttr activationZeroPointAttr;
  DenseElementsAttr weightDataAttr;
  DenseElementsAttr weightScaleAttr;
  DenseElementsAttr weightZeroPointAttr;
  DenseElementsAttr biasDataAttr;
  DenseElementsAttr biasScaleAttr;
  DenseElementsAttr biasZeroPointAttr;
  if (failed(getDenseConstant(activationDQ->getOperand(1), "activation scale",
                              activationScaleAttr, error)) ||
      failed(getDenseConstant(activationDQ->getOperand(2),
                              "activation zero point", activationZeroPointAttr,
                              error)) ||
      failed(getDenseConstant(weightDQ->getOperand(0), "quantized weight",
                              weightDataAttr, error)) ||
      failed(getDenseConstant(weightDQ->getOperand(1), "weight scale",
                              weightScaleAttr, error)) ||
      failed(getDenseConstant(weightDQ->getOperand(2), "weight zero point",
                              weightZeroPointAttr, error)) ||
      failed(getDenseConstant(biasDQ->getOperand(0), "quantized bias",
                              biasDataAttr, error)) ||
      failed(getDenseConstant(biasDQ->getOperand(1), "bias scale",
                              biasScaleAttr, error)) ||
      failed(getDenseConstant(biasDQ->getOperand(2), "bias zero point",
                              biasZeroPointAttr, error)))
    return failure();

  if (!conv->getResult(0).hasOneUse())
    return fail(error, "onnx.Conv result must have exactly one use");
  Operation *next = *conv->getResult(0).getUsers().begin();
  if (hasName(next, "onnx.Relu")) {
    if (next->getNumResults() != 1 || !next->getResult(0).hasOneUse())
      return fail(error, "onnx.Relu result must have exactly one use");
    next = *next->getResult(0).getUsers().begin();
  }
  if (!hasName(next, "onnx.QuantizeLinear") || next->getNumOperands() != 3 ||
      next->getNumResults() != 1)
    return fail(
        error,
        "onnx.Conv must be followed by optional Relu and QuantizeLinear");
  if (failed(requireTensor(next->getResult(0), 4, isUI8, "quantized output",
                           error)))
    return failure();

  DenseElementsAttr outputScaleAttr;
  DenseElementsAttr outputZeroPointAttr;
  if (failed(getDenseConstant(next->getOperand(1), "output scale",
                              outputScaleAttr, error)) ||
      failed(getDenseConstant(next->getOperand(2), "output zero point",
                              outputZeroPointAttr, error)))
    return failure();

  if (failed(readScalarFloat(activationScaleAttr, "activation scale",
                             params.activationScale, error)) ||
      failed(readScalarUI8(activationZeroPointAttr, "activation zero point",
                           params.activationZeroPoint, error)) ||
      failed(readFloatVector(weightScaleAttr, "weight scale", outputChannels,
                             params.weightScales, error)) ||
      failed(readIntVector(weightZeroPointAttr, "weight zero point",
                           outputChannels, isI8, params.weightZeroPoints,
                           error)) ||
      failed(readScalarFloat(outputScaleAttr, "output scale",
                             params.outputScale, error)) ||
      failed(readScalarUI8(outputZeroPointAttr, "output zero point",
                           params.outputZeroPoint, error)) ||
      failed(readFloatVector(biasScaleAttr, "bias scale", outputChannels,
                             params.biasScales, error)) ||
      failed(readIntVector(biasZeroPointAttr, "bias zero point", outputChannels,
                           isI32, params.biasZeroPoints, error)))
    return failure();

  for (int64_t zeroPoint : params.biasZeroPoints)
    if (zeroPoint != 0)
      return fail(error, "bias zero points must be zero");

  for (auto [channel, scales] :
       llvm::enumerate(llvm::zip(params.weightScales, params.biasScales))) {
    float expected =
        static_cast<float>(static_cast<float>(params.activationScale) *
                           static_cast<float>(std::get<0>(scales)));
    float actual = static_cast<float>(std::get<1>(scales));
    double tolerance =
        std::max(1.0e-12, std::abs(static_cast<double>(expected)) * 2.0e-6);
    if (std::abs(static_cast<double>(actual) - expected) > tolerance) {
      error = "bias scale at channel " + std::to_string(channel) +
              " does not match activation_scale * weight_scale";
      return failure();
    }
  }

  return success();
}

static LogicalResult discoverQAddChain(Operation *add, QAddChain &chain,
                                       std::string &error) {
  if (!hasName(add, "onnx.Add"))
    return fail(error, "target must be onnx.Add");
  if (add->getNumOperands() != 2 || add->getNumResults() != 1)
    return fail(error, "onnx.Add must have two operands and one result");

  Operation *lhsDQ;
  Operation *rhsDQ;
  if (failed(getDQ(add->getOperand(0), "lhs", lhsDQ, error)) ||
      failed(getDQ(add->getOperand(1), "rhs", rhsDQ, error)))
    return failure();
  if (lhsDQ->getNumResults() != 1 || rhsDQ->getNumResults() != 1)
    return fail(error, "input DequantizeLinear must have one result");

  auto lhsType = dyn_cast<RankedTensorType>(lhsDQ->getOperand(0).getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhsDQ->getOperand(0).getType());
  if (!lhsType || lhsType.getRank() != 4 || !lhsType.hasStaticShape() ||
      !isUI8(lhsType.getElementType()))
    return fail(error, "quantized lhs must be a static rank-4 NCHW ui8 tensor");
  if (rhsType != lhsType)
    return fail(error, "quantized rhs must have the quantized lhs tensor type");

  auto lhsFloatType = dyn_cast<RankedTensorType>(lhsDQ->getResult(0).getType());
  auto rhsFloatType = dyn_cast<RankedTensorType>(rhsDQ->getResult(0).getType());
  if (!lhsFloatType || lhsFloatType.getRank() != 4 ||
      !lhsFloatType.hasStaticShape() || !isF32(lhsFloatType.getElementType()))
    return fail(error,
                "dequantized lhs must be a static rank-4 NCHW f32 tensor");
  if (lhsFloatType.getShape() != lhsType.getShape())
    return fail(error, "dequantized lhs shape must match quantized lhs shape");
  if (rhsFloatType != lhsFloatType ||
      add->getResult(0).getType() != lhsFloatType)
    return fail(error, "Add operands and result must have one f32 tensor type");

  if (!add->getResult(0).hasOneUse())
    return fail(error, "onnx.Add result must have exactly one use");

  Operation *next = *add->getResult(0).getUsers().begin();
  Operation *relu = nullptr;
  if (hasName(next, "onnx.Relu")) {
    relu = next;
    if (relu->getNumOperands() != 1 || relu->getNumResults() != 1 ||
        relu->getOperand(0).getDefiningOp() != add ||
        relu->getResult(0).getType() != lhsFloatType ||
        !relu->getResult(0).hasOneUse())
      return fail(error, "onnx.Relu must be the single consumer of Add");
    next = *relu->getResult(0).getUsers().begin();
  }

  if (!hasName(next, "onnx.QuantizeLinear") || next->getNumOperands() != 3 ||
      next->getNumResults() != 1)
    return fail(
        error, "onnx.Add must be followed by optional Relu and QuantizeLinear");

  auto outputType = dyn_cast<RankedTensorType>(next->getResult(0).getType());
  if (outputType != lhsType)
    return fail(error,
                "quantized output must have the quantized input tensor type");

  chain = {lhsDQ, rhsDQ, add, relu, next};
  return success();
}

static LogicalResult extractQAddQuantParams(Operation *add,
                                            QAddQuantParams &params,
                                            std::string &error) {
  QAddChain chain;
  if (failed(discoverQAddChain(add, chain, error)))
    return failure();

  DenseElementsAttr lhsScaleAttr;
  DenseElementsAttr lhsZeroPointAttr;
  DenseElementsAttr rhsScaleAttr;
  DenseElementsAttr rhsZeroPointAttr;
  DenseElementsAttr outputScaleAttr;
  DenseElementsAttr outputZeroPointAttr;
  if (failed(getDenseConstant(chain.lhsDQ->getOperand(1), "lhs scale",
                              lhsScaleAttr, error)) ||
      failed(getDenseConstant(chain.lhsDQ->getOperand(2), "lhs zero point",
                              lhsZeroPointAttr, error)) ||
      failed(getDenseConstant(chain.rhsDQ->getOperand(1), "rhs scale",
                              rhsScaleAttr, error)) ||
      failed(getDenseConstant(chain.rhsDQ->getOperand(2), "rhs zero point",
                              rhsZeroPointAttr, error)) ||
      failed(getDenseConstant(chain.quantize->getOperand(1), "output scale",
                              outputScaleAttr, error)) ||
      failed(getDenseConstant(chain.quantize->getOperand(2),
                              "output zero point", outputZeroPointAttr, error)))
    return failure();

  if (failed(
          readScalarFloat(lhsScaleAttr, "lhs scale", params.lhsScale, error)) ||
      failed(readScalarUI8(lhsZeroPointAttr, "lhs zero point",
                           params.lhsZeroPoint, error)) ||
      failed(
          readScalarFloat(rhsScaleAttr, "rhs scale", params.rhsScale, error)) ||
      failed(readScalarUI8(rhsZeroPointAttr, "rhs zero point",
                           params.rhsZeroPoint, error)) ||
      failed(readScalarFloat(outputScaleAttr, "output scale",
                             params.outputScale, error)) ||
      failed(readScalarUI8(outputZeroPointAttr, "output zero point",
                           params.outputZeroPoint, error)))
    return failure();

  return success();
}

static bool roundToNearestEven(double value, int64_t &result) {
  constexpr double maxMultiplier =
      static_cast<double>(std::numeric_limits<int32_t>::max());

  if (!std::isfinite(value) || value < 0.0 || value > maxMultiplier + 0.5)
    return false;

  double lowerDouble = std::floor(value);
  int64_t lower = static_cast<int64_t>(lowerDouble);
  double fraction = value - lowerDouble;
  result = lower;
  if (fraction > 0.5 || (fraction == 0.5 && (lower & 1) != 0))
    ++result;

  return result <= std::numeric_limits<int32_t>::max();
}

static LogicalResult validateFixedPointApproximation(double realMultiplier,
                                                      int64_t multiplier,
                                                      int64_t shift,
                                                      std::string &error) {
  double step = std::ldexp(1.0, -shift);
  double approximation = static_cast<double>(multiplier) * step;
  double errorBound = 0.5 * step;
  if (std::abs(realMultiplier - approximation) > errorBound + 1.0e-15)
    return fail(error, "fixed-point approximation exceeds half an LSB");

  return success();
}

static LogicalResult computeFixedPointParams(double realMultiplier,
                                             int64_t &multiplier,
                                             int64_t &shift,
                                             std::string &error) {
  if (!std::isfinite(realMultiplier) || realMultiplier < 0.0)
    return fail(error, "real multiplier must be non-negative and finite");

  constexpr int64_t maxShift = 31;
  for (int64_t candidateShift = maxShift; candidateShift >= 0;
       --candidateShift) {

    double scaled = std::ldexp(realMultiplier, candidateShift);
    int64_t candidateMultiplier;
    if (!roundToNearestEven(scaled, candidateMultiplier))
      continue;

    if (failed(validateFixedPointApproximation(
            realMultiplier, candidateMultiplier, candidateShift, error)))
      return failure();

    multiplier = candidateMultiplier;
    shift = candidateShift;
    return success();
  }

  return fail(error, "real multiplier does not fit signed i32");
}

static LogicalResult computeConvRequantFixedPoint(
    double activationScale, ArrayRef<double> weightScales,
    ArrayRef<int64_t> weightZeroPoints, double outputScale,
    ConvRequantFixedPointParams &fixedPointParams, std::string &error) {

  if (!std::isfinite(activationScale) || activationScale <= 0.0)
    return fail(error, "activation scale must be positive and finite");

  if (!std::isfinite(outputScale) || outputScale <= 0.0)
    return fail(error, "output scale must be positive and finite");

  if (weightScales.size() != weightZeroPoints.size())
    return fail(error,
                "weight scale and zero point arrays must have equal lengths");

  for (auto [channel, zeroPoint] : llvm::enumerate(weightZeroPoints)) {
    if (zeroPoint != 0) {
      error = "weight zero point at channel " + std::to_string(channel) +
              " must be zero";
      return failure();
    }
  }

  for (auto [channel, weightScale] : llvm::enumerate(weightScales)) {

    if (!std::isfinite(weightScale) || weightScale <= 0.0) {
      error = "weight scale at channel " + std::to_string(channel) +
              " must be positive and finite";
      return failure();
    }

    double realMultiplier = activationScale * weightScale / outputScale;
    int64_t multiplier = 0;
    int64_t shift = 0;
    std::string coefficientError;
    if (failed(computeFixedPointParams(realMultiplier, multiplier, shift,
                                       coefficientError))) {
      error = "fixed-point coefficient at channel " + std::to_string(channel) +
              ": " + coefficientError;
      return failure();
    }

    fixedPointParams.multipliers.push_back(multiplier);
    fixedPointParams.shifts.push_back(shift);
  }

  return success();
}

static LogicalResult
computeQAddFixedPoint(double lhsScale, double rhsScale, double outputScale,
                      QAddFixedPointParams &fixedPointParams,
                      std::string &error) {
  if (!std::isfinite(lhsScale) || lhsScale <= 0.0)
    return fail(error, "lhs scale must be positive and finite");
  if (!std::isfinite(rhsScale) || rhsScale <= 0.0)
    return fail(error, "rhs scale must be positive and finite");
  if (!std::isfinite(outputScale) || outputScale <= 0.0)
    return fail(error, "output scale must be positive and finite");

  double lhsRealMultiplier = lhsScale / outputScale;
  double rhsRealMultiplier = rhsScale / outputScale;
  constexpr int64_t maxShift = 31;
  for (int64_t candidateShift = maxShift; candidateShift >= 0;
       --candidateShift) {
    int64_t lhsMultiplier;
    int64_t rhsMultiplier;
    if (!roundToNearestEven(std::ldexp(lhsRealMultiplier, candidateShift),
                            lhsMultiplier) ||
        !roundToNearestEven(std::ldexp(rhsRealMultiplier, candidateShift),
                            rhsMultiplier))
      continue;

    std::string approximationError;
    if (failed(validateFixedPointApproximation(lhsRealMultiplier, lhsMultiplier,
                                               candidateShift,
                                               approximationError))) {
      error = "lhs " + approximationError;
      return failure();
    }
    if (failed(validateFixedPointApproximation(rhsRealMultiplier, rhsMultiplier,
                                               candidateShift,
                                               approximationError))) {
      error = "rhs " + approximationError;
      return failure();
    }

    fixedPointParams = {lhsMultiplier, rhsMultiplier, candidateShift};
    return success();
  }

  return fail(error, "scale ratios do not fit signed i32 multipliers");
}

static LogicalResult readI64ArrayAttribute(Operation *op, StringRef name,
                                           ArrayRef<int64_t> defaultValues,
                                           size_t expectedSize,
                                           SmallVectorImpl<int64_t> &result,
                                           std::string &error) {
  Attribute attribute = op->getAttr(name);
  if (!attribute) {
    result.append(defaultValues.begin(), defaultValues.end());
    return success();
  }

  if (auto dense = dyn_cast<DenseI64ArrayAttr>(attribute)) {
    result.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
  } else if (auto array = dyn_cast<ArrayAttr>(attribute)) {
    for (Attribute element : array) {
      auto integer = dyn_cast<IntegerAttr>(element);
      if (!integer)
        return fail(error, (name + " must contain integer values").str());
      result.push_back(integer.getValue().getSExtValue());
    }
  } else {
    return fail(error, (name + " has an unsupported attribute type").str());
  }

  if (result.size() != expectedSize) {
    error = name.str() + " must contain " + std::to_string(expectedSize) +
            " values";
    return failure();
  }
  return success();
}

static LogicalResult prepareConvAccumulateSpec(Operation *conv,
                                               ConvAccumulateSpec &spec,
                                               std::string &error) {
  if (!hasName(conv, "onnx.Conv"))
    return fail(error, "Conv target must be onnx.Conv");

  if (conv->getNumOperands() != 3 || conv->getNumResults() != 1)
    return fail(error,
                "onnx.Conv must have activation, weight and bias operands");

  ConvQuantParams quantParams;
  if (failed(extractConvQuantParams(conv, quantParams, error)))
    return failure();

  if (llvm::any_of(quantParams.weightZeroPoints,
                   [](int64_t value) { return value != 0; }))
    return fail(error,
                "weight zero points must be zero for integer accumulation");

  Operation *activationDQ;
  Operation *weightDQ;
  Operation *biasDQ;
  if (failed(getDQ(conv->getOperand(0), "activation", activationDQ, error)) ||
      failed(getDQ(conv->getOperand(1), "weight", weightDQ, error)) ||
      failed(getDQ(conv->getOperand(2), "bias", biasDQ, error)))
    return failure();

  Value input = activationDQ->getOperand(0);
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || !inputType.hasStaticShape() || inputType.getRank() != 4 ||
      !isUI8(inputType.getElementType()))
    return fail(error, "quantized activation must be static rank-4 NCHW ui8");

  Value weight = weightDQ->getOperand(0);
  auto weightType = dyn_cast<RankedTensorType>(weight.getType());
  if (!weightType || !weightType.hasStaticShape() ||
      weightType.getRank() != 4 || !isI8(weightType.getElementType()))
    return fail(error, "quantized weight must be static rank-4 OIHW i8");

  Value bias = biasDQ->getOperand(0);
  auto biasType = dyn_cast<RankedTensorType>(bias.getType());
  if (!biasType || !biasType.hasStaticShape() || biasType.getRank() != 1 ||
      !isI32(biasType.getElementType()))
    return fail(error, "quantized bias must be static rank-1 i32");

  auto convOutputType =
      dyn_cast<RankedTensorType>(conv->getResult(0).getType());
  if (!convOutputType || !convOutputType.hasStaticShape() ||
      convOutputType.getRank() != 4)
    return fail(error, "onnx.Conv result must be a static rank-4 NCHW tensor");

  int64_t group = 1;
  if (auto groupAttr = conv->getAttrOfType<IntegerAttr>("group"))
    group = groupAttr.getValue().getSExtValue();
  if (group != 1)
    return fail(error, "onnx.Conv group must be 1");

  if (auto autoPad = conv->getAttrOfType<StringAttr>("auto_pad"))
    if (autoPad.getValue() != "NOTSET")
      return fail(error, "onnx.Conv auto_pad must be NOTSET");

  SmallVector<int64_t> strides;
  if (failed(
          readI64ArrayAttribute(conv, "strides", {1, 1}, 2, strides, error))) {
    return failure();
  }
  if (llvm::any_of(strides, [](int64_t value) { return value <= 0; }))
    return fail(error, "onnx.Conv strides must be positive");

  SmallVector<int64_t> pads;
  if (failed(
          readI64ArrayAttribute(conv, "pads", {0, 0, 0, 0}, 4, pads, error))) {
    return failure();
  }
  if (llvm::any_of(pads, [](int64_t value) { return value < 0; }))
    return fail(error, "onnx.Conv pads must be non-negative");

  SmallVector<int64_t> dilations;
  if (failed(readI64ArrayAttribute(conv, "dilations", {1, 1}, 2, dilations,
                                   error))) {
    return failure();
  }
  if (llvm::any_of(dilations, [](int64_t value) { return value <= 0; }))
    return fail(error, "onnx.Conv dilations must be positive");

  if (Attribute kernelShapeAttr = conv->getAttr("kernel_shape")) {
    SmallVector<int64_t> kernelShape;
    if (failed(readI64ArrayAttribute(conv, "kernel_shape", {}, 2, kernelShape,
                                     error)))
      return failure();

    if (kernelShape[0] != weightType.getDimSize(2) ||
        kernelShape[1] != weightType.getDimSize(3))
      return fail(error, "onnx.Conv kernel_shape must match weight shape");
  }

  if (weightType.getDimSize(1) != inputType.getDimSize(1))
    return fail(error, "weight input channels must match activation channels");
  if (biasType.getDimSize(0) != weightType.getDimSize(0))
    return fail(error, "bias length must match weight output channels");

  int64_t effectiveKernelHeight =
      dilations[0] * (weightType.getDimSize(2) - 1) + 1;
  int64_t effectiveKernelWidth =
      dilations[1] * (weightType.getDimSize(3) - 1) + 1;
  int64_t heightNumerator =
      inputType.getDimSize(2) + pads[0] + pads[2] - effectiveKernelHeight;
  int64_t widthNumerator =
      inputType.getDimSize(3) + pads[1] + pads[3] - effectiveKernelWidth;
  if (heightNumerator < 0 || widthNumerator < 0)
    return fail(error, "kernel is larger than the padded activation");

  SmallVector<int64_t> expectedOutputShape = {
      inputType.getDimSize(0), weightType.getDimSize(0),
      heightNumerator / strides[0] + 1, widthNumerator / strides[1] + 1};
  if (convOutputType.getShape() != ArrayRef<int64_t>(expectedOutputShape))
    return fail(error, "onnx.Conv result shape does not match its attributes");

  spec = {input,
          weight,
          bias,
          RankedTensorType::get(expectedOutputShape,
                                IntegerType::get(conv->getContext(), 32)),
          quantParams.activationZeroPoint,
          std::move(strides),
          std::move(pads),
          std::move(dilations),
          group};
  return success();
}

static LogicalResult validateQuantize(Operation *quantize,
                                      RankedTensorType &outputType,
                                      std::string &error) {
  if (!hasName(quantize, "onnx.QuantizeLinear"))
    return fail(error, "target must be onnx.QuantizeLinear");

  if (quantize->getNumOperands() != 3 || quantize->getNumResults() != 1)
    return fail(error,
                "onnx.QuantizeLinear must have three operands and one result");

  outputType = dyn_cast<RankedTensorType>(quantize->getResult(0).getType());
  if (!outputType || outputType.getRank() != 4 ||
      !outputType.hasStaticShape() || !isUI8(outputType.getElementType()))
    return fail(
        error, "QuantizeLinear result must be a static rank-4 NCHW ui8 tensor");

  return success();
}

static LogicalResult validateConvRequantChain(Operation *conv, Operation *relu,
                                              Operation *quantize,
                                              std::string &error) {
  if (!hasName(conv, "onnx.Conv"))
    return fail(error, "Conv target must be onnx.Conv");

  if (relu) {
    if (!hasName(relu, "onnx.Relu") || relu->getNumOperands() != 1 ||
        relu->getOperand(0).getDefiningOp() != conv)
      return fail(error, "Relu target must be the consumer of Conv");

    if (quantize->getOperand(0).getDefiningOp() != relu)
      return fail(error, "QuantizeLinear must consume the selected Relu");

  } else if (quantize->getOperand(0).getDefiningOp() != conv) {
    return fail(error, "QuantizeLinear must consume the selected Conv");
  }

  return success();
}

static LogicalResult
prepareConvRequantParams(Operation *quantize, RankedTensorType outputType,
                         Attribute multiplierParam, Attribute shiftParam,
                         Attribute outputZeroPointParam,
                         ConvRequantParams &params, std::string &error) {
  auto multipliers = dyn_cast<DenseI64ArrayAttr>(multiplierParam);
  auto shifts = dyn_cast<DenseI64ArrayAttr>(shiftParam);
  auto outputZeroPoint = dyn_cast<IntegerAttr>(outputZeroPointParam);
  if (!multipliers || !shifts || !outputZeroPoint)
    return fail(error, "requantization parameters have unsupported types");

  int64_t outputChannels = outputType.getDimSize(1);
  if (static_cast<int64_t>(multipliers.size()) != outputChannels ||
      static_cast<int64_t>(shifts.size()) != outputChannels)
    return fail(error,
                "multiplier and shift counts must equal output channels");

  for (auto [channel, multiplier] : llvm::enumerate(multipliers.asArrayRef())) {
    if (multiplier < 0 || multiplier > std::numeric_limits<int32_t>::max()) {
      error = "multiplier at channel " + std::to_string(channel) +
              " does not fit non-negative i32";
      return failure();
    }
  }

  for (auto [channel, shift] : llvm::enumerate(shifts.asArrayRef())) {
    if (shift < 0 || shift > 63) {
      error =
          "shift at channel " + std::to_string(channel) + " is outside [0, 63]";
      return failure();
    }
  }

  int64_t zeroPoint = outputZeroPoint.getValue().getSExtValue();
  if (zeroPoint < 0 || zeroPoint > 255)
    return fail(error, "output zero point is outside [0, 255]");

  DenseElementsAttr graphZeroPointAttr;
  int64_t graphZeroPoint;
  if (failed(getDenseConstant(quantize->getOperand(2), "output zero point",
                              graphZeroPointAttr, error)) ||
      failed(readScalarUI8(graphZeroPointAttr, "output zero point",
                           graphZeroPoint, error)))
    return failure();

  if (zeroPoint != graphZeroPoint)
    return fail(error,
                "output zero point parameter does not match QuantizeLinear");

  params = {multipliers, shifts, zeroPoint};
  return success();
}

static LogicalResult
prepareConvRequantRewrite(Operation *conv, Operation *relu, Operation *quantize,
                          Attribute multiplierParam, Attribute shiftParam,
                          Attribute outputZeroPointParam,
                          ConvRequantRewrite &rewrite, std::string &error) {
  RankedTensorType outputType;
  if (failed(validateQuantize(quantize, outputType, error)))
    return failure();

  if (failed(validateConvRequantChain(conv, relu, quantize, error)))
    return failure();

  ConvAccumulateSpec accumulate;
  if (failed(prepareConvAccumulateSpec(conv, accumulate, error)))
    return failure();
  if (accumulate.resultType.getShape() != outputType.getShape())
    return fail(error,
                "integer accumulator and QuantizeLinear shapes must match");

  ConvRequantParams params{};
  if (failed(prepareConvRequantParams(quantize, outputType, multiplierParam,
                                      shiftParam, outputZeroPointParam, params,
                                      error)))
    return failure();

  rewrite = {conv,
             relu,
             quantize,
             std::move(accumulate),
             params.multipliers,
             params.shifts,
             outputType,
             params.outputZeroPoint};
  return success();
}

static Value createI32TensorConstant(TransformRewriter &rewriter, Location loc,
                                     DenseI64ArrayAttr values) {
  SmallVector<int32_t> narrowed;
  narrowed.reserve(values.size());
  for (int64_t value : values.asArrayRef())
    narrowed.push_back(static_cast<int32_t>(value));

  auto type = RankedTensorType::get({static_cast<int64_t>(narrowed.size())},
                                    rewriter.getI32Type());
  auto value = DenseIntElementsAttr::get(type, narrowed);
  return arith::ConstantOp::create(rewriter, loc, type, value);
}

static Operation *createConvAccumulateOp(TransformRewriter &rewriter,
                                         const ConvRequantRewrite &rewrite) {
  Location loc = rewrite.conv->getLoc();
  rewriter.setInsertionPoint(rewrite.conv);

  OperationState state(loc, standalone::ConvAccumulateOp::getOperationName());
  state.addOperands({rewrite.accumulate.input, rewrite.accumulate.weight,
                     rewrite.accumulate.bias});
  state.addTypes(rewrite.accumulate.resultType);
  state.addAttribute(
      "input_zero_point",
      rewriter.getI32IntegerAttr(rewrite.accumulate.inputZeroPoint));
  state.addAttribute("strides",
                     DenseI64ArrayAttr::get(rewriter.getContext(),
                                            rewrite.accumulate.strides));
  state.addAttribute("pads", DenseI64ArrayAttr::get(rewriter.getContext(),
                                                    rewrite.accumulate.pads));
  state.addAttribute("dilations",
                     DenseI64ArrayAttr::get(rewriter.getContext(),
                                            rewrite.accumulate.dilations));
  state.addAttribute("group",
                     rewriter.getI64IntegerAttr(rewrite.accumulate.group));

  return rewriter.create(state);
}

static Operation *createConvRequantOp(TransformRewriter &rewriter,
                                      const ConvRequantRewrite &rewrite,
                                      Value accumulator, bool hasRelu) {
  Location loc = rewrite.quantize->getLoc();
  rewriter.setInsertionPoint(rewrite.quantize);

  Value multipliers =
      createI32TensorConstant(rewriter, loc, rewrite.multipliers);
  Value shifts = createI32TensorConstant(rewriter, loc, rewrite.shifts);
  Value output = tensor::EmptyOp::create(
      rewriter, loc, rewrite.outputType.getShape(),
      rewrite.outputType.getElementType(), rewrite.outputType.getEncoding());

  OperationState state(loc, standalone::ConvRequantOp::getOperationName());
  state.addOperands({accumulator, multipliers, shifts, output});
  state.addTypes(rewrite.outputType);
  state.addAttribute("output_zero_point",
                     rewriter.getI32IntegerAttr(rewrite.outputZeroPoint));
  state.addAttribute("relu_enable",
                     rewriter.getI32IntegerAttr(hasRelu ? 1 : 0));

  return rewriter.create(state);
}

static LogicalResult readIntegerParam(Attribute attribute, StringRef label,
                                      int64_t minimum, int64_t maximum,
                                      int64_t &result, std::string &error) {
  auto integer = dyn_cast<IntegerAttr>(attribute);
  if (!integer)
    return fail(error, (label + " parameter must be an integer").str());

  result = integer.getValue().getSExtValue();
  if (result < minimum || result > maximum) {
    error = label.str() + " parameter is outside [" + std::to_string(minimum) +
            ", " + std::to_string(maximum) + "]";
    return failure();
  }
  return success();
}

static LogicalResult
prepareQAddRewrite(Operation *lhsDQ, Operation *rhsDQ, Operation *add,
                   Operation *relu, Operation *quantize,
                   Attribute lhsMultiplierParam, Attribute rhsMultiplierParam,
                   Attribute shiftParam, Attribute lhsZeroPointParam,
                   Attribute rhsZeroPointParam, Attribute outputZeroPointParam,
                   bool hasRelu, QAddRewrite &rewrite, std::string &error) {
  QAddChain chain;
  if (failed(discoverQAddChain(add, chain, error)))
    return failure();

  if (chain.lhsDQ != lhsDQ || chain.rhsDQ != rhsDQ)
    return fail(error, "DequantizeLinear handles do not match Add operands");
  if (chain.quantize != quantize)
    return fail(error, "QuantizeLinear handle does not match the Add chain");
  if (hasRelu ? chain.relu != relu : chain.relu != nullptr)
    return fail(error, "optional Relu handle does not match the Add chain");

  int64_t lhsMultiplier;
  int64_t rhsMultiplier;
  int64_t shift;
  int64_t lhsZeroPoint;
  int64_t rhsZeroPoint;
  int64_t outputZeroPoint;
  if (failed(readIntegerParam(lhsMultiplierParam, "lhs multiplier", 0,
                              std::numeric_limits<int32_t>::max(),
                              lhsMultiplier, error)) ||
      failed(readIntegerParam(rhsMultiplierParam, "rhs multiplier", 0,
                              std::numeric_limits<int32_t>::max(),
                              rhsMultiplier, error)) ||
      failed(readIntegerParam(shiftParam, "shift", 0, 63, shift, error)) ||
      failed(readIntegerParam(lhsZeroPointParam, "lhs zero point", 0, 255,
                              lhsZeroPoint, error)) ||
      failed(readIntegerParam(rhsZeroPointParam, "rhs zero point", 0, 255,
                              rhsZeroPoint, error)) ||
      failed(readIntegerParam(outputZeroPointParam, "output zero point", 0, 255,
                              outputZeroPoint, error)))
    return failure();

  QAddQuantParams graphParams;
  if (failed(extractQAddQuantParams(add, graphParams, error)))
    return failure();
  if (lhsZeroPoint != graphParams.lhsZeroPoint ||
      rhsZeroPoint != graphParams.rhsZeroPoint ||
      outputZeroPoint != graphParams.outputZeroPoint)
    return fail(error, "zero point parameters do not match the ONNX QDQ chain");

  QAddFixedPointParams expectedFixedPoint;
  if (failed(computeQAddFixedPoint(graphParams.lhsScale, graphParams.rhsScale,
                                   graphParams.outputScale, expectedFixedPoint,
                                   error)))
    return failure();
  if (lhsMultiplier != expectedFixedPoint.lhsMultiplier ||
      rhsMultiplier != expectedFixedPoint.rhsMultiplier ||
      shift != expectedFixedPoint.shift)
    return fail(error,
                "fixed-point parameters do not match the ONNX QDQ scales");

  auto outputType =
      cast<RankedTensorType>(chain.quantize->getResult(0).getType());
  rewrite = {chain,        outputType,   expectedFixedPoint,
             lhsZeroPoint, rhsZeroPoint, outputZeroPoint};
  return success();
}

static Operation *createQAddOp(TransformRewriter &rewriter,
                               const QAddRewrite &rewrite, bool hasRelu) {
  Location loc = rewrite.chain.quantize->getLoc();
  rewriter.setInsertionPoint(rewrite.chain.quantize);

  Value output = tensor::EmptyOp::create(
      rewriter, loc, rewrite.outputType.getShape(),
      rewrite.outputType.getElementType(), rewrite.outputType.getEncoding());

  OperationState state(loc, standalone::QAddReluOp::getOperationName());
  state.addOperands({rewrite.chain.lhsDQ->getOperand(0),
                     rewrite.chain.rhsDQ->getOperand(0), output});
  state.addTypes(rewrite.outputType);
  state.addAttribute("lhs_multiplier", rewriter.getI32IntegerAttr(
                                           rewrite.fixedPoint.lhsMultiplier));
  state.addAttribute("rhs_multiplier", rewriter.getI32IntegerAttr(
                                           rewrite.fixedPoint.rhsMultiplier));
  state.addAttribute("shift",
                     rewriter.getI32IntegerAttr(rewrite.fixedPoint.shift));
  state.addAttribute("lhs_zero_point",
                     rewriter.getI32IntegerAttr(rewrite.lhsZeroPoint));
  state.addAttribute("rhs_zero_point",
                     rewriter.getI32IntegerAttr(rewrite.rhsZeroPoint));
  state.addAttribute("output_zero_point",
                     rewriter.getI32IntegerAttr(rewrite.outputZeroPoint));
  state.addAttribute("relu_enable",
                     rewriter.getI32IntegerAttr(hasRelu ? 1 : 0));
  return rewriter.create(state);
}

static LogicalResult
lowerONNXQAdd(TransformRewriter &rewriter, ArrayRef<Operation *> lhsDQOps,
              ArrayRef<Operation *> rhsDQOps, ArrayRef<Operation *> addOps,
              ArrayRef<Operation *> reluOps, ArrayRef<Operation *> quantizeOps,
              ArrayRef<Attribute> lhsMultiplierParams,
              ArrayRef<Attribute> rhsMultiplierParams,
              ArrayRef<Attribute> shiftParams,
              ArrayRef<Attribute> lhsZeroPointParams,
              ArrayRef<Attribute> rhsZeroPointParams,
              ArrayRef<Attribute> outputZeroPointParams, bool hasRelu,
              SmallVectorImpl<Operation *> &newQAdds, std::string &error) {
  size_t rewriteCount = addOps.size();
  if (lhsDQOps.size() != rewriteCount || rhsDQOps.size() != rewriteCount ||
      quantizeOps.size() != rewriteCount ||
      (hasRelu && reluOps.size() != rewriteCount) ||
      (!hasRelu && !reluOps.empty()) ||
      lhsMultiplierParams.size() != rewriteCount ||
      rhsMultiplierParams.size() != rewriteCount ||
      shiftParams.size() != rewriteCount ||
      lhsZeroPointParams.size() != rewriteCount ||
      rhsZeroPointParams.size() != rewriteCount ||
      outputZeroPointParams.size() != rewriteCount)
    return fail(error, "QAdd operation and parameter handles must have equal "
                       "lengths");

  SmallVector<QAddRewrite, 1> rewrites;
  rewrites.reserve(rewriteCount);
  for (size_t index = 0; index < rewriteCount; ++index) {
    QAddRewrite rewrite;
    if (failed(prepareQAddRewrite(
            lhsDQOps[index], rhsDQOps[index], addOps[index],
            hasRelu ? reluOps[index] : nullptr, quantizeOps[index],
            lhsMultiplierParams[index], rhsMultiplierParams[index],
            shiftParams[index], lhsZeroPointParams[index],
            rhsZeroPointParams[index], outputZeroPointParams[index], hasRelu,
            rewrite, error))) {
      error = "QAdd at index " + std::to_string(index) + ": " + error;
      return failure();
    }
    rewrites.push_back(rewrite);
  }

  newQAdds.reserve(newQAdds.size() + rewriteCount);
  for (const QAddRewrite &rewrite : rewrites) {
    Operation *qadd = createQAddOp(rewriter, rewrite, hasRelu);
    rewriter.replaceOp(rewrite.chain.quantize, qadd->getResults());
    if (rewrite.chain.relu)
      rewriter.eraseOp(rewrite.chain.relu);
    rewriter.eraseOp(rewrite.chain.add);

    if (rewrite.chain.lhsDQ == rewrite.chain.rhsDQ) {
      if (rewrite.chain.lhsDQ->use_empty())
        rewriter.eraseOp(rewrite.chain.lhsDQ);
    } else {
      if (rewrite.chain.lhsDQ->use_empty())
        rewriter.eraseOp(rewrite.chain.lhsDQ);
      if (rewrite.chain.rhsDQ->use_empty())
        rewriter.eraseOp(rewrite.chain.rhsDQ);
    }
    newQAdds.push_back(qadd);
  }

  return success();
}

class StandaloneTransformDialectExtension
    : public TransformDialectExtension<StandaloneTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      StandaloneTransformDialectExtension)

  using Base::Base;

  void init() {
    declareDependentDialect<arith::ArithDialect>();
    declareDependentDialect<standalone::StandaloneDialect>();
    declareDependentDialect<tensor::TensorDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "Standalone/StandaloneTransformOps.cpp.inc"
        >();
  }
};

} // namespace

void transform::ExtractONNXQAddQuantParamsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getAddMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure transform::ExtractONNXQAddQuantParamsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results, TransformState &state) {
  (void)rewriter;
  (void)state;

  QAddQuantParams params;
  std::string error;
  if (failed(extractQAddQuantParams(target, params, error)))
    return emitSilenceableFailure(getLoc(), error);

  MLIRContext *context = getContext();
  results.push_back(FloatAttr::get(Float64Type::get(context), params.lhsScale));
  results.push_back(
      IntegerAttr::get(IntegerType::get(context, 64), params.lhsZeroPoint));
  results.push_back(FloatAttr::get(Float64Type::get(context), params.rhsScale));
  results.push_back(
      IntegerAttr::get(IntegerType::get(context, 64), params.rhsZeroPoint));
  results.push_back(
      FloatAttr::get(Float64Type::get(context), params.outputScale));
  results.push_back(
      IntegerAttr::get(IntegerType::get(context, 64), params.outputZeroPoint));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform::ComputeQAddFixedPointOp::apply(TransformRewriter &rewriter,
                                          TransformResults &results,
                                          TransformState &state) {
  (void)rewriter;

  ArrayRef<Attribute> lhsScales = state.getParams(getLhsScale());
  ArrayRef<Attribute> rhsScales = state.getParams(getRhsScale());
  ArrayRef<Attribute> outputScales = state.getParams(getOutputScale());
  size_t parameterCount = lhsScales.size();
  if (rhsScales.size() != parameterCount ||
      outputScales.size() != parameterCount)
    return emitSilenceableFailure(
        getLoc(), "QAdd scale parameter handles must have equal lengths");

  MLIRContext *context = getContext();
  Type i64 = IntegerType::get(context, 64);
  SmallVector<Attribute> lhsMultiplierResults;
  SmallVector<Attribute> rhsMultiplierResults;
  SmallVector<Attribute> shiftResults;
  lhsMultiplierResults.reserve(parameterCount);
  rhsMultiplierResults.reserve(parameterCount);
  shiftResults.reserve(parameterCount);

  for (size_t index = 0; index < parameterCount; ++index) {
    auto lhsScale = dyn_cast<FloatAttr>(lhsScales[index]);
    auto rhsScale = dyn_cast<FloatAttr>(rhsScales[index]);
    auto outputScale = dyn_cast<FloatAttr>(outputScales[index]);
    if (!lhsScale || !rhsScale || !outputScale) {
      std::string error = "QAdd scale set at index " + std::to_string(index) +
                          " has unsupported types";
      return emitSilenceableFailure(getLoc(), error);
    }

    QAddFixedPointParams fixedPoint;
    std::string error;
    if (failed(computeQAddFixedPoint(
            lhsScale.getValueAsDouble(), rhsScale.getValueAsDouble(),
            outputScale.getValueAsDouble(), fixedPoint, error))) {
      error = "QAdd scale set at index " + std::to_string(index) + ": " + error;
      return emitSilenceableFailure(getLoc(), error);
    }

    lhsMultiplierResults.push_back(
        IntegerAttr::get(i64, fixedPoint.lhsMultiplier));
    rhsMultiplierResults.push_back(
        IntegerAttr::get(i64, fixedPoint.rhsMultiplier));
    shiftResults.push_back(IntegerAttr::get(i64, fixedPoint.shift));
  }

  results.setParams(cast<OpResult>(getLhsMultiplier()), lhsMultiplierResults);
  results.setParams(cast<OpResult>(getRhsMultiplier()), rhsMultiplierResults);
  results.setParams(cast<OpResult>(getShift()), shiftResults);
  return DiagnosedSilenceableFailure::success();
}

void transform::ExtractONNXConvQuantParamsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getConvMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure transform::ExtractONNXConvQuantParamsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results, TransformState &state) {
  (void)rewriter;
  (void)state;

  ConvQuantParams params;
  std::string error;
  if (failed(extractConvQuantParams(target, params, error)))
    return emitSilenceableFailure(getLoc(), error);

  MLIRContext *context = getContext();
  results.push_back(
      FloatAttr::get(Float64Type::get(context), params.activationScale));
  results.push_back(IntegerAttr::get(IntegerType::get(context, 64),
                                     params.activationZeroPoint));
  results.push_back(DenseF64ArrayAttr::get(context, params.weightScales));
  results.push_back(DenseI64ArrayAttr::get(context, params.weightZeroPoints));
  results.push_back(
      FloatAttr::get(Float64Type::get(context), params.outputScale));
  results.push_back(
      IntegerAttr::get(IntegerType::get(context, 64), params.outputZeroPoint));
  results.push_back(DenseF64ArrayAttr::get(context, params.biasScales));
  results.push_back(DenseI64ArrayAttr::get(context, params.biasZeroPoints));
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
transform::ComputeConvRequantFixedPointOp::apply(TransformRewriter &rewriter,
                                                 TransformResults &results,
                                                 TransformState &state) {

  (void)rewriter;

  ArrayRef<Attribute> activationScales = state.getParams(getActivationScale());
  ArrayRef<Attribute> weightScales = state.getParams(getWeightScales());
  ArrayRef<Attribute> weightZeroPoints = state.getParams(getWeightZeroPoints());
  ArrayRef<Attribute> outputScales = state.getParams(getOutputScale());

  size_t parameterCount = activationScales.size();
  if (weightScales.size() != parameterCount ||
      weightZeroPoints.size() != parameterCount ||
      outputScales.size() != parameterCount)
    return emitSilenceableFailure(
        getLoc(), "fixed-point parameter handles must have equal lengths");

  MLIRContext *context = getContext();
  SmallVector<Attribute> multiplierResults;
  SmallVector<Attribute> shiftResults;
  multiplierResults.reserve(parameterCount);
  shiftResults.reserve(parameterCount);

  for (size_t index = 0; index < parameterCount; ++index) {
    auto activationScale = dyn_cast<FloatAttr>(activationScales[index]);
    auto chScales = dyn_cast<DenseF64ArrayAttr>(weightScales[index]);
    auto outputScale = dyn_cast<FloatAttr>(outputScales[index]);
    auto chZeroPoints = dyn_cast<DenseI64ArrayAttr>(weightZeroPoints[index]);

    if (!activationScale || !chScales || !chZeroPoints || !outputScale) {
      std::string error = "fixed-point parameter set at index " +
                          std::to_string(index) + " has unsupported types";
      return emitSilenceableFailure(getLoc(), error);
    }

    std::string error;
    ConvRequantFixedPointParams fixedPointParams;
    if (failed(computeConvRequantFixedPoint(
            activationScale.getValueAsDouble(), chScales.asArrayRef(),
            chZeroPoints.asArrayRef(), outputScale.getValueAsDouble(),
            fixedPointParams, error))) {
      error = "fixed-point parameter set at index " + std::to_string(index) +
              ": " + error;
      return emitSilenceableFailure(getLoc(), error);
    }

    multiplierResults.push_back(
        DenseI64ArrayAttr::get(context, fixedPointParams.multipliers));
    shiftResults.push_back(
        DenseI64ArrayAttr::get(context, fixedPointParams.shifts));
  }

  results.setParams(cast<OpResult>(getMultipliers()), multiplierResults);
  results.setParams(cast<OpResult>(getShifts()), shiftResults);

  return DiagnosedSilenceableFailure::success();
}

void transform::LowerONNXConvRequantOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getConvMutable(), effects);
  consumesHandle(getQuantizeMutable(), effects);
  onlyReadsHandle(getMultipliersMutable(), effects);
  onlyReadsHandle(getShiftsMutable(), effects);
  onlyReadsHandle(getOutputZeroPointMutable(), effects);
  producesHandle(getOperation()->getResults(), effects);
  modifiesPayload(effects);
}

static LogicalResult lowerONNXConvRequant(
    TransformRewriter &rewriter, ArrayRef<Operation *> convOps,
    ArrayRef<Operation *> reluOps, ArrayRef<Operation *> quantizeOps,
    ArrayRef<Attribute> multiplierParams, ArrayRef<Attribute> shiftParams,
    ArrayRef<Attribute> outputZeroPointParams, bool hasRelu,
    SmallVectorImpl<Operation *> &newAccumulates,
    SmallVectorImpl<Operation *> &newRequants, std::string &error) {

  size_t rewriteCount = convOps.size();
  if (quantizeOps.size() != rewriteCount ||
      (hasRelu && reluOps.size() != rewriteCount) ||
      (!hasRelu && !reluOps.empty()) ||
      multiplierParams.size() != rewriteCount ||
      shiftParams.size() != rewriteCount ||
      outputZeroPointParams.size() != rewriteCount)
    return fail(error, "Conv, optional Relu, QuantizeLinear and parameter "
                       "handles must have equal lengths");

  SmallVector<ConvRequantRewrite, 1> rewrites;
  rewrites.reserve(rewriteCount);
  for (size_t index = 0; index < rewriteCount; ++index) {
    ConvRequantRewrite rewrite;

    if (failed(prepareConvRequantRewrite(
            convOps[index], hasRelu ? reluOps[index] : nullptr,
            quantizeOps[index], multiplierParams[index], shiftParams[index],
            outputZeroPointParams[index], rewrite, error))) {
      error = "requantization at index " + std::to_string(index) + ": " + error;
      return failure();
    }

    rewrites.push_back(rewrite);
  }

  newAccumulates.reserve(newAccumulates.size() + rewriteCount);
  newRequants.reserve(newRequants.size() + rewriteCount);
  for (const ConvRequantRewrite &rewrite : rewrites) {
    Operation *accumulate = createConvAccumulateOp(rewriter, rewrite);
    Operation *requant = createConvRequantOp(rewriter, rewrite,
                                             accumulate->getResult(0), hasRelu);

    rewriter.replaceOp(rewrite.quantize, requant->getResults());
    if (rewrite.relu)
      rewriter.eraseOp(rewrite.relu);
    rewriter.eraseOp(rewrite.conv);

    newAccumulates.push_back(accumulate);
    newRequants.push_back(requant);
  }

  return success();
}

DiagnosedSilenceableFailure
transform::LowerONNXConvRequantOp::apply(TransformRewriter &rewriter,
                                         TransformResults &results,
                                         TransformState &state) {
  SmallVector<Operation *> convOps;
  for (Operation *op : state.getPayloadOps(getConv()))
    convOps.push_back(op);

  SmallVector<Operation *> quantizeOps;
  for (Operation *op : state.getPayloadOps(getQuantize()))
    quantizeOps.push_back(op);

  SmallVector<Operation *> accumulates;
  SmallVector<Operation *> requants;
  std::string error;
  if (failed(lowerONNXConvRequant(
          rewriter, convOps, {}, quantizeOps, state.getParams(getMultipliers()),
          state.getParams(getShifts()), state.getParams(getOutputZeroPoint()),
          false, accumulates, requants, error)))
    return emitSilenceableFailure(getLoc(), error);

  results.set(cast<OpResult>(getAccumulate()), accumulates);
  results.set(cast<OpResult>(getRequant()), requants);
  return DiagnosedSilenceableFailure::success();
}

void transform::LowerONNXConvReluRequantOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getConvMutable(), effects);
  consumesHandle(getReluMutable(), effects);
  consumesHandle(getQuantizeMutable(), effects);
  onlyReadsHandle(getMultipliersMutable(), effects);
  onlyReadsHandle(getShiftsMutable(), effects);
  onlyReadsHandle(getOutputZeroPointMutable(), effects);
  producesHandle(getOperation()->getResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform::LowerONNXConvReluRequantOp::apply(TransformRewriter &rewriter,
                                             TransformResults &results,
                                             TransformState &state) {
  SmallVector<Operation *> convOps;
  for (Operation *op : state.getPayloadOps(getConv()))
    convOps.push_back(op);

  SmallVector<Operation *> reluOps;
  for (Operation *op : state.getPayloadOps(getRelu()))
    reluOps.push_back(op);

  SmallVector<Operation *> quantizeOps;
  for (Operation *op : state.getPayloadOps(getQuantize()))
    quantizeOps.push_back(op);

  SmallVector<Operation *> accumulates;
  SmallVector<Operation *> requants;
  std::string error;
  if (failed(lowerONNXConvRequant(rewriter, convOps, reluOps, quantizeOps,
                                  state.getParams(getMultipliers()),
                                  state.getParams(getShifts()),
                                  state.getParams(getOutputZeroPoint()), true,
                                  accumulates, requants, error)))
    return emitSilenceableFailure(getLoc(), error);

  results.set(cast<OpResult>(getAccumulate()), accumulates);
  results.set(cast<OpResult>(getRequant()), requants);

  return DiagnosedSilenceableFailure::success();
}

void transform::LowerONNXQAddOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getLhsDqMutable(), effects);
  consumesHandle(getRhsDqMutable(), effects);
  consumesHandle(getAddMutable(), effects);
  consumesHandle(getQuantizeMutable(), effects);
  onlyReadsHandle(getLhsMultiplierMutable(), effects);
  onlyReadsHandle(getRhsMultiplierMutable(), effects);
  onlyReadsHandle(getShiftMutable(), effects);
  onlyReadsHandle(getLhsZeroPointMutable(), effects);
  onlyReadsHandle(getRhsZeroPointMutable(), effects);
  onlyReadsHandle(getOutputZeroPointMutable(), effects);
  producesHandle(getOperation()->getResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform::LowerONNXQAddOp::apply(TransformRewriter &rewriter,
                                  TransformResults &results,
                                  TransformState &state) {
  SmallVector<Operation *> lhsDQOps;
  for (Operation *op : state.getPayloadOps(getLhsDq()))
    lhsDQOps.push_back(op);
  SmallVector<Operation *> rhsDQOps;
  for (Operation *op : state.getPayloadOps(getRhsDq()))
    rhsDQOps.push_back(op);
  SmallVector<Operation *> addOps;
  for (Operation *op : state.getPayloadOps(getAdd()))
    addOps.push_back(op);
  SmallVector<Operation *> quantizeOps;
  for (Operation *op : state.getPayloadOps(getQuantize()))
    quantizeOps.push_back(op);

  SmallVector<Operation *> qadds;
  std::string error;
  if (failed(lowerONNXQAdd(
          rewriter, lhsDQOps, rhsDQOps, addOps, {}, quantizeOps,
          state.getParams(getLhsMultiplier()),
          state.getParams(getRhsMultiplier()), state.getParams(getShift()),
          state.getParams(getLhsZeroPoint()),
          state.getParams(getRhsZeroPoint()),
          state.getParams(getOutputZeroPoint()), false, qadds, error)))
    return emitSilenceableFailure(getLoc(), error);

  results.set(cast<OpResult>(getQadd()), qadds);
  return DiagnosedSilenceableFailure::success();
}

void transform::LowerONNXQAddReluOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getLhsDqMutable(), effects);
  consumesHandle(getRhsDqMutable(), effects);
  consumesHandle(getAddMutable(), effects);
  consumesHandle(getReluMutable(), effects);
  consumesHandle(getQuantizeMutable(), effects);
  onlyReadsHandle(getLhsMultiplierMutable(), effects);
  onlyReadsHandle(getRhsMultiplierMutable(), effects);
  onlyReadsHandle(getShiftMutable(), effects);
  onlyReadsHandle(getLhsZeroPointMutable(), effects);
  onlyReadsHandle(getRhsZeroPointMutable(), effects);
  onlyReadsHandle(getOutputZeroPointMutable(), effects);
  producesHandle(getOperation()->getResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform::LowerONNXQAddReluOp::apply(TransformRewriter &rewriter,
                                      TransformResults &results,
                                      TransformState &state) {
  SmallVector<Operation *> lhsDQOps;
  for (Operation *op : state.getPayloadOps(getLhsDq()))
    lhsDQOps.push_back(op);
  SmallVector<Operation *> rhsDQOps;
  for (Operation *op : state.getPayloadOps(getRhsDq()))
    rhsDQOps.push_back(op);
  SmallVector<Operation *> addOps;
  for (Operation *op : state.getPayloadOps(getAdd()))
    addOps.push_back(op);
  SmallVector<Operation *> reluOps;
  for (Operation *op : state.getPayloadOps(getRelu()))
    reluOps.push_back(op);
  SmallVector<Operation *> quantizeOps;
  for (Operation *op : state.getPayloadOps(getQuantize()))
    quantizeOps.push_back(op);

  SmallVector<Operation *> qadds;
  std::string error;
  if (failed(lowerONNXQAdd(
          rewriter, lhsDQOps, rhsDQOps, addOps, reluOps, quantizeOps,
          state.getParams(getLhsMultiplier()),
          state.getParams(getRhsMultiplier()), state.getParams(getShift()),
          state.getParams(getLhsZeroPoint()),
          state.getParams(getRhsZeroPoint()),
          state.getParams(getOutputZeroPoint()), true, qadds, error)))
    return emitSilenceableFailure(getLoc(), error);

  results.set(cast<OpResult>(getQadd()), qadds);
  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "Standalone/StandaloneTransformOps.cpp.inc"

void mlir::standalone::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<StandaloneTransformDialectExtension>();
}
