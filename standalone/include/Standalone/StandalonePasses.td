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

#endif // STANDALONE_PASS
