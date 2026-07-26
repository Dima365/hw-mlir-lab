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
    : Standalone_Op<"qadd_relu"> {
  let summary = "8x8 quantized residual add with optional ReLU";
  let description = [{
    Adds two per-tensor uint8 activation tiles. Each input is centered by its
    zero point and scaled into a shared fixed-point domain. The wide sum is
    rounded once, optionally clamped by ReLU, and saturated into the output
    per-tensor uint8 domain.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$lhs,
    Arg<AnyMemRef, "", [MemRead]>:$rhs,
    Arg<AnyMemRef, "", [MemWrite]>:$out,
    I32Attr:$lhs_multiplier,
    I32Attr:$rhs_multiplier,
    I32Attr:$shift,
    I32Attr:$lhs_zero_point,
    I32Attr:$rhs_zero_point,
    I32Attr:$output_zero_point,
    I32Attr:$relu_enable
  );

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $out `:` type($out) `)`
    attr-dict
  }];

  let hasVerifier = 1;
}

def Standalone_ConvRequantOp
    : Standalone_Op<"conv_requant"> {
  let summary = "8x8 per-output-channel Conv requantization";
  let description = [{
    Requantizes an 8x8 tile of final i32 Conv accumulators to a per-tensor
    uint8 activation domain. Rows are spatial positions and columns are output
    channels. The multiplier and shift memrefs contain one fixed-point pair per
    output-channel column.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$acc,
    Arg<AnyMemRef, "", [MemRead]>:$multiplier,
    Arg<AnyMemRef, "", [MemRead]>:$shift,
    Arg<AnyMemRef, "", [MemWrite]>:$out,
    I32Attr:$output_zero_point,
    I32Attr:$relu_enable
  );

  let assemblyFormat = [{
    `ins` `(` $acc `,` $multiplier `,` $shift `:`
      type($acc) `,` type($multiplier) `,` type($shift) `)`
    `outs` `(` $out `:` type($out) `)`
    attr-dict
  }];

  let hasVerifier = 1;
}

#endif // STANDALONE_OPS
