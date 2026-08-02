//===- StandaloneTransformOps.td - Standalone transform ops ---*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_TRANSFORM_OPS
#define STANDALONE_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ExtractONNXConvQuantParamsOp
    : TransformDialectOp<"standalone.extract_onnx_conv_qparams", [
        TransformOpInterface,
        TransformEachOpTrait,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Extract quantization parameters around an ONNX QDQ Conv";
  let description = [{
    Reads a handle to `onnx.Conv` operations and follows their QDQ dataflow.
    The supported payload shape is three `onnx.DequantizeLinear` inputs for
    activation, weight and bias, followed by the Conv, optional `onnx.Relu`,
    and `onnx.QuantizeLinear`.

    Activation and output quantization must be per-tensor uint8. Weight and
    bias quantization must be per-output-channel with axis zero. All numeric
    parameters and the quantized weight and bias tensors must be
    `onnx.Constant` operations.

    The operation validates positive finite scales, zero bias zero-points and
    `bias_scale[c] = activation_scale * weight_scale[c]` within f32 tolerance.
    It only reads the payload and returns one parameter of each kind per Conv.
    A contract violation produces a silenceable failure.
  }];

  let arguments = (ins TransformHandleTypeInterface:$conv);
  let results = (outs
      TransformParamTypeInterface:$activation_scale,
      TransformParamTypeInterface:$activation_zero_point,
      TransformParamTypeInterface:$weight_scales,
      TransformParamTypeInterface:$weight_zero_points,
      TransformParamTypeInterface:$output_scale,
      TransformParamTypeInterface:$output_zero_point,
      TransformParamTypeInterface:$bias_scales,
      TransformParamTypeInterface:$bias_zero_points);

  let assemblyFormat =
      "$conv attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // STANDALONE_TRANSFORM_OPS
