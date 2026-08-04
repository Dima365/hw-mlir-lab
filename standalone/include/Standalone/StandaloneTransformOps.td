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

def ComputeConvRequantFixedPointOp
    : TransformDialectOp<"standalone.compute_conv_requant_fixed_point", [
        DeclareOpInterfaceMethods<TransformOpInterface>,
        MemoryEffectsOpInterface,
        ParamProducerTransformOpTrait
      ]> {
  let summary = "Compute fixed-point parameters for CONV_REQUANT";
  let description = [{
    Reads parameter handles produced by `extract_onnx_conv_qparams`. For every
    output channel, it approximates

    ```
    activation_scale * weight_scale[channel] / output_scale
    ```

    as `multiplier[channel] / 2^shift[channel]`. Multipliers are non-negative
    signed-i32 values and shifts are selected from 31 down to zero to maximize
    precision. Coefficient rounding uses round-to-nearest, ties-to-even.

    Each operand may contain parameters for several Conv operations. Operand
    lists are matched by index and must have equal lengths. The returned lists
    contain one multiplier array and one shift array for each Conv. A later
    lowering slices each array into the eight-channel tiles consumed by
    `standalone.conv_requant`. Weight zero-points must be zero until asymmetric
    weight correction is implemented in the accumulator path.
  }];

  let arguments = (ins
      TransformParamTypeInterface:$activation_scale,
      TransformParamTypeInterface:$weight_scales,
      TransformParamTypeInterface:$weight_zero_points,
      TransformParamTypeInterface:$output_scale);
  let results = (outs
      TransformParamTypeInterface:$multipliers,
      TransformParamTypeInterface:$shifts);

  let assemblyFormat =
      "$activation_scale `,` $weight_scales `,` $weight_zero_points `,` "
      "$output_scale attr-dict `:` functional-type(operands, results)";
}

#endif // STANDALONE_TRANSFORM_OPS
