// RUN: standalone-opt %s | FileCheck %s --check-prefix=PARSE
// RUN: standalone-opt --lower-systolic-u8s8-to-func-call %s | FileCheck %s --check-prefix=LOWER

module {
  func.func @matmul(
      %lhs: memref<8x8xui8>,
      %rhs: memref<8x8xi8>,
      %acc: memref<8x8xi32>) {
    standalone.systolic_u8s8_matmul
      ins(%lhs, %rhs : memref<8x8xui8>, memref<8x8xi8>)
      outs(%acc : memref<8x8xi32>)
      {lhs_zero_point = 113 : i32}
    return
  }
}

// PARSE: standalone.systolic_u8s8_matmul
// PARSE-SAME: lhs_zero_point = 113 : i32

// LOWER: func.func private @systolic_u8s8_matmul_8x8
// LOWER: %[[ZP:.*]] = arith.constant 113 : i32
// LOWER: call @systolic_u8s8_matmul_8x8
// LOWER-SAME: %[[ZP]])
