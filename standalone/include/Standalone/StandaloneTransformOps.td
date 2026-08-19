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

def LowerONNXConvRequantOp
    : TransformDialectOp<"standalone.lower_onnx_conv_requant", [
        DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Lower ONNX QDQ Conv to integer standalone operations";
  let description = [{
    Replaces an `onnx.Conv -> onnx.QuantizeLinear` QDQ chain with
    `standalone.conv_accumulate -> standalone.conv_requant`. Raw quantized
    activation, weight and bias values are taken from the three
    `onnx.DequantizeLinear` inputs of Conv.

    Multipliers, shifts and the output zero point are parameter handles
    produced by `compute_conv_requant_fixed_point` and
    `extract_onnx_conv_qparams`. The transform materializes the per-channel
    arrays as `arith.constant` tensors, creates a `tensor.empty` destination
    and replaces the QuantizeLinear result. The old Conv and QuantizeLinear
    operations are erased.

    Conv, QuantizeLinear and parameter sets are paired by position. All
    handles must contain the same number of entries. Only static NCHW/OIHW,
    `group = 1` and explicit or default NOTSET padding are supported. This
    transform does not lower the new integer convolution to CPU code or to
    systolic-array tiles.
  }];

  let arguments = (ins
      TransformHandleTypeInterface:$conv,
      TransformHandleTypeInterface:$quantize,
      TransformParamTypeInterface:$multipliers,
      TransformParamTypeInterface:$shifts,
      TransformParamTypeInterface:$output_zero_point);
  let results = (outs
      TransformHandleTypeInterface:$accumulate,
      TransformHandleTypeInterface:$requant);

  let assemblyFormat =
      "$conv `,` $quantize `,` $multipliers `,` $shifts `,` "
      "$output_zero_point attr-dict `:` functional-type(operands, results)";
}

def LowerONNXConvReluRequantOp
    : TransformDialectOp<"standalone.lower_onnx_conv_relu_requant", [
        DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Lower ONNX QDQ Conv and ReLU to integer standalone operations";
  let description = [{
    Replaces an `onnx.Conv -> onnx.Relu -> onnx.QuantizeLinear` chain with
    `standalone.conv_accumulate -> standalone.conv_requant`. It has the same
    quantization and shape contract as `lower_onnx_conv_requant`, and sets
    `relu_enable = 1` on the requantization operation. Conv, ReLU and
    QuantizeLinear are consumed and erased by the rewrite.
  }];

  let arguments = (ins
      TransformHandleTypeInterface:$conv,
      TransformHandleTypeInterface:$relu,
      TransformHandleTypeInterface:$quantize,
      TransformParamTypeInterface:$multipliers,
      TransformParamTypeInterface:$shifts,
      TransformParamTypeInterface:$output_zero_point);
  let results = (outs
      TransformHandleTypeInterface:$accumulate,
      TransformHandleTypeInterface:$requant);

  let assemblyFormat =
      "$conv `,` $relu `,` $quantize `,` $multipliers `,` $shifts `,` "
      "$output_zero_point attr-dict `:` functional-type(operands, results)";
}

def ExtractONNXQAddQuantParamsOp
    : TransformDialectOp<"standalone.extract_onnx_qadd_qparams", [
        TransformOpInterface,
        TransformEachOpTrait,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Extract quantization parameters around an ONNX residual Add";
  let description = [{
    Reads a handle to `onnx.Add` operations whose two operands are produced by
    scalar per-tensor `onnx.DequantizeLinear` operations and whose result is
    consumed by optional `onnx.Relu` and `onnx.QuantizeLinear` operations.

    Both quantized inputs and the quantized output must be static rank-4 NCHW
    uint8 tensors with identical shapes. All scales and zero points must be
    scalar `onnx.Constant` values. The operation only reads the payload and
    returns one parameter of each kind per Add.
  }];

  let arguments = (ins TransformHandleTypeInterface:$add);
  let results = (outs
      TransformParamTypeInterface:$lhs_scale,
      TransformParamTypeInterface:$lhs_zero_point,
      TransformParamTypeInterface:$rhs_scale,
      TransformParamTypeInterface:$rhs_zero_point,
      TransformParamTypeInterface:$output_scale,
      TransformParamTypeInterface:$output_zero_point);

  let assemblyFormat =
      "$add attr-dict `:` functional-type(operands, results)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ComputeQAddFixedPointOp
    : TransformDialectOp<"standalone.compute_qadd_fixed_point", [
        DeclareOpInterfaceMethods<TransformOpInterface>,
        MemoryEffectsOpInterface,
        ParamProducerTransformOpTrait
      ]> {
  let summary = "Compute shared fixed-point parameters for QADD_RELU";
  let description = [{
    Approximates `lhs_scale / output_scale` and
    `rhs_scale / output_scale` as two non-negative signed-i32 multipliers with
    one shared binary shift. Candidate shifts are considered from 31 down to
    zero and coefficient rounding uses round-to-nearest, ties-to-even.

    Operand lists are paired by position and must have equal lengths. The
    operation produces one scalar lhs multiplier, rhs multiplier and shift per
    residual Add.
  }];

  let arguments = (ins
      TransformParamTypeInterface:$lhs_scale,
      TransformParamTypeInterface:$rhs_scale,
      TransformParamTypeInterface:$output_scale);
  let results = (outs
      TransformParamTypeInterface:$lhs_multiplier,
      TransformParamTypeInterface:$rhs_multiplier,
      TransformParamTypeInterface:$shift);

  let assemblyFormat =
      "$lhs_scale `,` $rhs_scale `,` $output_scale attr-dict `:` "
      "functional-type(operands, results)";
}

def LowerONNXQAddOp
    : TransformDialectOp<"standalone.lower_onnx_qadd", [
        DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Lower an ONNX QDQ Add to standalone.qadd_relu";
  let description = [{
    Replaces `DequantizeLinear + DequantizeLinear -> Add -> QuantizeLinear`
    with tensor-form `standalone.qadd_relu` and `relu_enable = 0`. The new
    operation consumes the original uint8 values before dequantization, so an
    upstream Conv requantization boundary remains intact.
  }];

  let arguments = (ins
      TransformHandleTypeInterface:$lhs_dq,
      TransformHandleTypeInterface:$rhs_dq,
      TransformHandleTypeInterface:$add,
      TransformHandleTypeInterface:$quantize,
      TransformParamTypeInterface:$lhs_multiplier,
      TransformParamTypeInterface:$rhs_multiplier,
      TransformParamTypeInterface:$shift,
      TransformParamTypeInterface:$lhs_zero_point,
      TransformParamTypeInterface:$rhs_zero_point,
      TransformParamTypeInterface:$output_zero_point);
  let results = (outs TransformHandleTypeInterface:$qadd);

  let assemblyFormat =
      "$lhs_dq `,` $rhs_dq `,` $add `,` $quantize `,` $lhs_multiplier `,` "
      "$rhs_multiplier `,` $shift `,` $lhs_zero_point `,` $rhs_zero_point `,` "
      "$output_zero_point attr-dict `:` functional-type(operands, results)";
}

def LowerONNXQAddReluOp
    : TransformDialectOp<"standalone.lower_onnx_qadd_relu", [
        DeclareOpInterfaceMethods<TransformOpInterface>,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Lower an ONNX QDQ Add and ReLU to standalone.qadd_relu";
  let description = [{
    Replaces
    `DequantizeLinear + DequantizeLinear -> Add -> Relu -> QuantizeLinear`
    with tensor-form `standalone.qadd_relu` and `relu_enable = 1`. It otherwise
    has the same quantization and shape contract as `lower_onnx_qadd`.
  }];

  let arguments = (ins
      TransformHandleTypeInterface:$lhs_dq,
      TransformHandleTypeInterface:$rhs_dq,
      TransformHandleTypeInterface:$add,
      TransformHandleTypeInterface:$relu,
      TransformHandleTypeInterface:$quantize,
      TransformParamTypeInterface:$lhs_multiplier,
      TransformParamTypeInterface:$rhs_multiplier,
      TransformParamTypeInterface:$shift,
      TransformParamTypeInterface:$lhs_zero_point,
      TransformParamTypeInterface:$rhs_zero_point,
      TransformParamTypeInterface:$output_zero_point);
  let results = (outs TransformHandleTypeInterface:$qadd);

  let assemblyFormat =
      "$lhs_dq `,` $rhs_dq `,` $add `,` $relu `,` $quantize `,` "
      "$lhs_multiplier `,` $rhs_multiplier `,` $shift `,` $lhs_zero_point `,` "
      "$rhs_zero_point `,` $output_zero_point attr-dict `:` "
      "functional-type(operands, results)";
}

#endif // STANDALONE_TRANSFORM_OPS
