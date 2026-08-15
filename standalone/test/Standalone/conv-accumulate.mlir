// RUN: standalone-opt %s | FileCheck %s

module {
  func.func @conv_accumulate(
      %input: tensor<1x3x5x7xui8>,
      %weight: tensor<8x3x3x3xi8>,
      %bias: tensor<8xi32>) -> tensor<1x8x3x4xi32> {
    %acc = standalone.conv_accumulate
      ins(%input, %weight, %bias
          : tensor<1x3x5x7xui8>, tensor<8x3x3x3xi8>, tensor<8xi32>)
      {
        input_zero_point = 113 : i32,
        strides = array<i64: 2, 2>,
        pads = array<i64: 1, 1, 1, 1>,
        dilations = array<i64: 1, 1>,
        group = 1 : i64
      } -> tensor<1x8x3x4xi32>
    return %acc : tensor<1x8x3x4xi32>
  }
}

// CHECK: standalone.conv_accumulate
// CHECK-SAME: dilations = array<i64: 1, 1>
// CHECK-SAME: group = 1 : i64
// CHECK-SAME: input_zero_point = 113 : i32
// CHECK-SAME: pads = array<i64: 1, 1, 1, 1>
// CHECK-SAME: strides = array<i64: 2, 2>
// CHECK-SAME: -> tensor<1x8x3x4xi32>
