//===- StandaloneTransformOps.cpp - Standalone transform ops -------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneTransformOps.h"

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

class StandaloneTransformDialectExtension
    : public TransformDialectExtension<StandaloneTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      StandaloneTransformDialectExtension)

  using Base::Base;

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "Standalone/StandaloneTransformOps.cpp.inc"
        >();
  }
};

} // namespace

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

#define GET_OP_CLASSES
#include "Standalone/StandaloneTransformOps.cpp.inc"

void mlir::standalone::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<StandaloneTransformDialectExtension>();
}
