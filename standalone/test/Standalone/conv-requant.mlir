// RUN: standalone-opt %s | FileCheck %s --check-prefix=PARSE
// RUN: standalone-opt --lower-conv-requant-to-func-call %s | FileCheck %s --check-prefix=LOWER

module {
  func.func @requant(
      %acc: memref<8x8xi32>,
      %multiplier: memref<8xi32>,
      %shift: memref<8xi32>,
      %out: memref<8x8xi8>) {
    standalone.conv_requant
      ins(%acc, %multiplier, %shift
          : memref<8x8xi32>, memref<8xi32>, memref<8xi32>)
      outs(%out : memref<8x8xi8>)
      {output_zero_point = 17 : i32, relu_enable = 1 : i32}
    return
  }
}

// PARSE: standalone.conv_requant
// PARSE-SAME: output_zero_point = 17 : i32
// PARSE-SAME: relu_enable = 1 : i32

// LOWER: func.func private @conv_requant_8x8
// LOWER: %[[ZP:.*]] = arith.constant 17 : i32
// LOWER: %[[RELU:.*]] = arith.constant 1 : i32
// LOWER: call @conv_requant_8x8
// LOWER-SAME: %[[ZP]], %[[RELU]]
