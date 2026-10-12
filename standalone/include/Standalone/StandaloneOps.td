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
include "mlir/Interfaces/DestinationStyleOpInterface.td"
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

def Standalone_SystolicU8S8MatmulOp
    : Standalone_Op<"systolic_u8s8_matmul"> {
  let summary = "8x8 u8s8 systolic matrix-accumulate";
  let description = [{
    Computes one fixed-size matrix-accumulate tile:

    ```
    acc += (lhs - lhs_zero_point) * rhs
    ```

    `lhs` contains unsigned activations, `rhs` contains signed weights and
    `acc` is both the initial and final i32 accumulator. The operation matches
    the quantized mode of the RTL systolic array.
    Signless i8 or unsigned ui8 lhs storage is accepted; in both cases its
    bits are interpreted as uint8. Weight zero point is implicitly zero.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyMemRef, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemRead, MemWrite]>:$acc,
    I32Attr:$lhs_zero_point
  );

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $acc `:` type($acc) `)`
    attr-dict
  }];

  let hasVerifier = 1;
}

def Standalone_QAddReluOp
    : Standalone_Op<"qadd_relu", [
        DestinationStyleOpInterface,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Quantized residual add with optional ReLU";
  let description = [{
    Adds two per-tensor uint8 activations. Each input is centered by its zero
    point and scaled into a shared fixed-point domain. The wide sum is rounded
    once, optionally clamped by ReLU, and saturated into the output per-tensor
    uint8 domain.

    The tensor form represents a static NCHW tensor or a rank-2 [M, C]
    matrix (M = batch * height * width) in high-level IR
    and returns the output tensor. The buffer form is the hardware tile
    operation: it accepts two 8x8 input tiles, has no SSA result and writes the
    output memref.

    Tensor TilingInterface supports fixed-size tiles. Pad tails before tiling;
    the buffer lowering accepts rank-2 tensors only after tiling to 8x8.
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
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutMutable(); }
  }];
}


def Standalone_ConvRequantOp
    : Standalone_Op<"conv_requant", [
        DestinationStyleOpInterface,
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Per-output-channel Conv requantization";
  let description = [{
    Requantizes final i32 Conv accumulators to a per-tensor uint8 activation
    domain.

    The tensor form represents a static NCHW tensor or a rank-2 [M, C]
    matrix (M = batch * height * width). Both forms support TilingInterface;
    multiplier and shift are sliced only along the channel dimension (1).
    Tensor tiles must have static sizes; pad tails before tiling. The buffer
    lowering accepts rank-2 tensors only after tiling to 8x8.
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
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getOutMutable(); }
  }];
}

#endif // STANDALONE_OPS
