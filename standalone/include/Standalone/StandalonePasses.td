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

def LowerRequantizeToFuncCall
    : Pass<"lower-requantize-to-func-call", "mlir::ModuleOp"> {
  let summary = "Lower standalone.requantize to the epilogue runtime call";
  let description = [{
    Replaces standalone.requantize with a func.call to the epilogue_8x8 runtime
    function; the mult/shift/zero_point attributes become arith.constant operands.
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
