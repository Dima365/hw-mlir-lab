//===- StandalonePasses.td - Standalone dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_PASS
#define STANDALONE_PASS

include "mlir/Pass/PassBase.td"

def StandaloneSwitchBarFoo: Pass<"standalone-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

def ConvertLinalgMatmulToSystolic
    : Pass<"convert-linalg-matmul-to-systolic", "::mlir::ModuleOp"> {
  let summary = "Convert 8x8 linalg.matmul ops to standalone.systolic_matmul";
}

def LowerSystolicToFuncCall
    : Pass<"lower-systolic-to-func-call", "mlir::ModuleOp"> {
  let summary = "Lower standalone.systolic_matmul to simulator runtime call";
  let description = [{
    Replaces standalone.systolic_matmul memref operation with a func.call to
    systolic_matmul_8x8 runtime function.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}

def LowerSystolicU8S8ToFuncCall
    : Pass<"lower-systolic-u8s8-to-func-call", "mlir::ModuleOp"> {
  let summary = "Lower quantized u8s8 systolic tiles to the simulator runtime";
  let description = [{
    Replaces standalone.systolic_u8s8_matmul with a func.call to the
    systolic_u8s8_matmul_8x8 runtime function. The activation zero point
    attribute becomes an i32 call operand.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def PrepareU8S8Matmul : Pass<"prepare-u8s8-matmul", "mlir::ModuleOp"> {
  let summary = "Normalize and pad quantized im2col matrix multiplications";
  let description = [{
    After the fork's convert-conv2d-to-img2col, normalize the static
    input[B,K,P] / weight[F,K] / output[B,F,P] contraction to
    input[B*P,K] / weight[K,F] / output[B*P,F] using Linalg transposes
    and Tensor reshapes. Require unsigned i8 activations, signed i8 weights,
    i32 accumulation, a constant activation zero point and weight zero point 0.

    Use the standard Linalg padding utility for multiples of eight, with the
    activation zero point for input padding and zeros for weights/accumulators.
    Mark the resulting generic for Transform Dialect tiling. No im2col indexing
    or hardware tiling is implemented by this pass.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def FuseMatmulEpilogues : Pass<"fuse-matmul-epilogues", "mlir::ModuleOp"> {
  let summary = "Share output tiles between matmul, requant and optional QADD";
  let description = [{
    After prepare-u8s8-matmul and before reduction tiling, normalize supported
    NCHW epilogues to padded [batch * height * width, channels] matrices.
    Use SCF TilingInterface fusion to share 8x8 output loops. Fuse bias
    initialization, complete matmul reductions, per-channel requantization,
    and optionally QADD with either one or two requantized producers.
    Preserve the intermediate uint8 rounding and saturation and all ReLU
    flags. Keep im2col and the boundary between successive spatial convolutions.

    Only the exact prepared layout chain with single-use accumulators and
    a channel-broadcast bias is accepted. A requantized input to QADD must
    have a single use. Unsupported chains are errors; no independent NCHW
    epilogue scheduling is provided. Boundary activations may still feed
    both branches of the next residual block. All tensor epilogues must be
    rank-2 8x8 tiles after this pass.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::scf::SCFDialect",
    "mlir::standalone::StandaloneDialect"
  ];
}

def LowerConvRequantToTiles
    : Pass<"lower-conv-requant-to-tiles", "mlir::ModuleOp"> {
  let summary = "Convert fused Conv requantization tiles to hardware buffers";
  let description = [{
    Converts already tiled rank-2 8x8 standalone.conv_requant operations
    from fuse-matmul-epilogues to the hardware buffer form. Copies only the
    accumulator microtile and its eight per-output-channel multiplier/shift
    values at the synchronous runtime boundary. Adds no output tile loop.
    Full NCHW tensors and untiled matrices are rejected.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::standalone::StandaloneDialect"
  ];
}

def ConvertLinalgU8S8MatmulToSystolic
    : Pass<"convert-linalg-u8s8-matmul-to-systolic", "mlir::ModuleOp"> {
  let summary = "Convert exact 8x8 Linalg u8s8 matmul to the systolic op";
  let description = [{
    Replaces marked buffer-form linalg.generic operations implementing

      acc += (extui(lhs) - lhs_zero_point) * extsi(rhs)

    with standalone.systolic_u8s8_matmul. The pass accepts fixed 8x8 i8/ui8 lhs,
    i8 rhs and i32 accumulator buffers and requires a constant lhs zero point.
    Unsigned lhs arithmetic must be explicit in the Linalg body. The zero
    point may be a scalar input or a captured constant; an omitted subtraction
    denotes zero. Operations with extra memory effects are rejected.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::standalone::StandaloneDialect"
  ];
}

def LowerQAddReluToTiles
    : Pass<"lower-qadd-relu-to-tiles", "mlir::ModuleOp"> {
  let summary = "Convert fused QADD_RELU tiles to hardware buffers";
  let description = [{
    Converts already tiled rank-2 8x8 standalone.qadd_relu operations
    from fuse-matmul-epilogues to hardware buffers, preserving their
    position-by-channel ordering. Copies only the microtiles at the
    synchronous runtime boundary and adds no output tile loop.
    Full NCHW tensors and untiled matrices are rejected.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::standalone::StandaloneDialect"
  ];
}

def LowerQAddReluToFuncCall
    : Pass<"lower-qadd-relu-to-func-call", "mlir::ModuleOp"> {
  let summary = "Lower standalone.qadd_relu to its simulator runtime call";
  let description = [{
    Replaces standalone.qadd_relu with a func.call to the qadd_relu_8x8 runtime
    function. Quantization attributes become arith.constant operands.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

def LowerConvRequantToFuncCall
    : Pass<"lower-conv-requant-to-func-call", "mlir::ModuleOp"> {
  let summary = "Lower standalone.conv_requant to its simulator runtime call";
  let description = [{
    Replaces standalone.conv_requant with a func.call to the
    conv_requant_8x8 runtime function. Scalar attributes become arith.constant
    operands; per-channel multiplier and shift values remain memref operands.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

def CreateCInterfaceEntryWrappers
    : Pass<"create-c-interface-entry-wrappers", "mlir::ModuleOp"> {
  let summary = "Create C interface entry wrappers for public memref functions";
  let description = [{
    Creates <name>_entry wrappers for public memref functions of the form
    memref inputs -> one memref result. The wrapper takes the original inputs
    plus an explicit output memref, calls the original function, copies the
    result into the output, and is marked with llvm.emit_c_interface.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // STANDALONE_PASS
