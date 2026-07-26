// RUN: standalone-opt %s | FileCheck %s --check-prefix=PARSE
// RUN: standalone-opt --lower-qadd-relu-to-func-call %s | FileCheck %s --check-prefix=LOWER

module {
  func.func @qadd_relu(
      %lhs: memref<8x8xi8>,
      %rhs: memref<8x8xi8>,
      %out: memref<8x8xi8>) {
    standalone.qadd_relu
      ins(%lhs, %rhs : memref<8x8xi8>, memref<8x8xi8>)
      outs(%out : memref<8x8xi8>)
      {
        lhs_multiplier = 17 : i32,
        rhs_multiplier = 13 : i32,
        shift = 4 : i32,
        lhs_zero_point = 68 : i32,
        rhs_zero_point = 65 : i32,
        output_zero_point = 3 : i32,
        relu_enable = 1 : i32
      }
    return
  }
}

// PARSE: standalone.qadd_relu
// PARSE-SAME: lhs_multiplier = 17 : i32
// PARSE-SAME: lhs_zero_point = 68 : i32
// PARSE-SAME: output_zero_point = 3 : i32
// PARSE-SAME: relu_enable = 1 : i32
// PARSE-SAME: rhs_multiplier = 13 : i32
// PARSE-SAME: rhs_zero_point = 65 : i32
// PARSE-SAME: shift = 4 : i32

// LOWER: func.func private @qadd_relu_8x8
// LOWER: %[[LM:.*]] = arith.constant 17 : i32
// LOWER: %[[RM:.*]] = arith.constant 13 : i32
// LOWER: %[[SHIFT:.*]] = arith.constant 4 : i32
// LOWER: %[[LZP:.*]] = arith.constant 68 : i32
// LOWER: %[[RZP:.*]] = arith.constant 65 : i32
// LOWER: %[[OZP:.*]] = arith.constant 3 : i32
// LOWER: %[[RELU:.*]] = arith.constant 1 : i32
// LOWER: call @qadd_relu_8x8
// LOWER-SAME: %[[LM]], %[[RM]], %[[SHIFT]], %[[LZP]], %[[RZP]], %[[OZP]], %[[RELU]]
