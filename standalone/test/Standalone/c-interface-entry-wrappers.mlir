// RUN: standalone-opt --create-c-interface-entry-wrappers %s | FileCheck %s

module {
  func.func @single_input(%input: memref<4xui8>) -> memref<4xui8> {
    return %input : memref<4xui8>
  }

  func.func @two_inputs(%lhs: memref<4xui8>, %rhs: memref<4xui8>)
      -> memref<4xui8> {
    return %rhs : memref<4xui8>
  }

  func.func private @private_function(memref<4xui8>) -> memref<4xui8>

  func.func @scalar_result(%input: memref<4xui8>) -> i32 {
    %zero = arith.constant 0 : i32
    return %zero : i32
  }
}

// CHECK-LABEL: func.func @single_input_entry(
// CHECK-SAME: %[[INPUT:.*]]: memref<4xui8>, %[[OUTPUT:.*]]: memref<4xui8>)
// CHECK-SAME: attributes {llvm.emit_c_interface}
// CHECK: %[[RESULT:.*]] = call @single_input(%[[INPUT]])
// CHECK: memref.copy %[[RESULT]], %[[OUTPUT]]

// CHECK-LABEL: func.func @two_inputs_entry(
// CHECK-SAME: %[[LHS:.*]]: memref<4xui8>, %[[RHS:.*]]: memref<4xui8>, %[[OUTPUT:.*]]: memref<4xui8>)
// CHECK-SAME: attributes {llvm.emit_c_interface}
// CHECK: %[[RESULT:.*]] = call @two_inputs(%[[LHS]], %[[RHS]])
// CHECK: memref.copy %[[RESULT]], %[[OUTPUT]]

// CHECK-NOT: @private_function_entry
// CHECK-NOT: @scalar_result_entry
