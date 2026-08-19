//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "Standalone/StandaloneTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def Standalone_FooOp : Standalone_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Standalone_SystolicMatmulOp
    : Standalone_Op<"systolic_matmul"> {
  let summary = "8x8 systolic matmul on memrefs";

  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyMemRef, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemRead, MemWrite]>:$acc
  );

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $acc `:` type($acc) `)`
    attr-dict
  }];
}

def Standalone_QAddReluOp
    : Standalone_Op<"qadd_relu", [
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Quantized residual add with optional ReLU";
  let description = [{
    Adds two per-tensor uint8 activations. Each input is centered by its zero
    point and scaled into a shared fixed-point domain. The wide sum is rounded
    once, optionally clamped by ReLU, and saturated into the output per-tensor
    uint8 domain.

    The tensor form represents an entire static NCHW tensor in high-level IR
    and returns the output tensor. The buffer form is the hardware tile
    operation: it accepts two 8x8 input tiles, has no SSA result and writes the
    output memref.
  }];

  let arguments = (ins
    AnyShaped:$lhs,
    AnyShaped:$rhs,
    AnyShaped:$out,
    I32Attr:$lhs_multiplier,
    I32Attr:$rhs_multiplier,
    I32Attr:$shift,
    I32Attr:$lhs_zero_point,
    I32Attr:$rhs_zero_point,
    I32Attr:$output_zero_point,
    I32Attr:$relu_enable
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $out `:` type($out) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let hasVerifier = 1;
}

def Standalone_ConvAccumulateOp
    : Standalone_Op<"conv_accumulate", [Pure]> {
  let summary = "Integer convolution accumulation in NCHW layout";
  let description = [{
    Computes an unfused integer convolution accumulator:

    ```
    acc[n, oc, oh, ow] = bias[oc] +
      sum(ic, kh, kw) ((input[n, ic, ih, iw] - input_zero_point) *
                       weight[oc, ic, kh, kw])
    ```

    The first implementation accepts static rank-4 NCHW uint8 activations,
    static rank-4 OIHW signed-i8 weights, a rank-1 i32 bias and produces a
    static rank-4 NCHW i32 tensor. Only `group = 1` is supported. Padding is
    interpreted using the ONNX `[top, left, bottom, right]` convention; padded
    activation values equal `input_zero_point` and therefore add zero to the
    accumulator.

    This is a high-level accelerator operation. It intentionally has no CPU or
    systolic-array lowering yet.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    AnyRankedTensor:$bias,
    I32Attr:$input_zero_point,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$pads,
    DenseI64ArrayAttr:$dilations,
    I64Attr:$group
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `ins` `(` $input `,` $weight `,` $bias `:`
      type($input) `,` type($weight) `,` type($bias) `)`
    attr-dict `->` type($result)
  }];

  let hasVerifier = 1;
}

def Standalone_ConvRequantOp
    : Standalone_Op<"conv_requant", [
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Per-output-channel Conv requantization";
  let description = [{
    Requantizes final i32 Conv accumulators to a per-tensor uint8 activation
    domain.

    The tensor form represents an entire static NCHW tensor in high-level IR.
    Its rank-1 multiplier and shift tensors contain one fixed-point pair per
    output channel and the operation returns the output tensor.

    The buffer form is the hardware tile operation. It accepts an 8x8
    accumulator tile where rows are spatial positions and columns are output
    channels. Its multiplier and shift memrefs contain eight fixed-point pairs.
    The buffer form has no SSA result and writes the output memref.
  }];

  let arguments = (ins
    AnyShaped:$acc,
    AnyShaped:$multiplier,
    AnyShaped:$shift,
    AnyShaped:$out,
    I32Attr:$output_zero_point,
    I32Attr:$relu_enable
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    `ins` `(` $acc `,` $multiplier `,` $shift `:`
      type($acc) `,` type($multiplier) `,` type($shift) `)`
    `outs` `(` $out `:` type($out) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let hasVerifier = 1;
}

#endif // STANDALONE_OPS
