// RUN: standalone-opt --allow-unregistered-dialect \
// RUN:   --transform-preload-library="transform-library-paths=%S/../../../projects/resnet18/qadd_relu.mlir" \
// RUN:   --transform-interpreter %s | FileCheck %s

module {
  func.func @lower_qadd(
      %acc: tensor<1x8x2x4xi32>,
      %relu_rhs_q: tensor<1x8x2x4xui8>,
      %direct_lhs_q: tensor<1x8x2x4xui8>,
      %direct_rhs_q: tensor<1x8x2x4xui8>)
      -> (tensor<1x8x2x4xui8>, tensor<1x8x2x4xui8>) {
    %conv_multiplier = arith.constant dense<1> : tensor<8xi32>
    %conv_shift = arith.constant dense<0> : tensor<8xi32>
    %conv_empty = tensor.empty() : tensor<1x8x2x4xui8>
    %conv_q = standalone.conv_requant
      ins(%acc, %conv_multiplier, %conv_shift
          : tensor<1x8x2x4xi32>, tensor<8xi32>, tensor<8xi32>)
      outs(%conv_empty : tensor<1x8x2x4xui8>)
      {
        output_zero_point = 3 : i32,
        relu_enable = 0 : i32
      } -> tensor<1x8x2x4xui8>

    %relu_lhs_scale = "onnx.Constant"() {
      value = dense<0.125> : tensor<f32>
    } : () -> tensor<f32>
    %relu_lhs_zp = "onnx.Constant"() {
      value = dense<3> : tensor<ui8>
    } : () -> tensor<ui8>
    %relu_lhs = "onnx.DequantizeLinear"(
        %conv_q, %relu_lhs_scale, %relu_lhs_zp) {axis = 1 : si64}
        : (tensor<1x8x2x4xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xf32>
    %relu_rhs_scale = "onnx.Constant"() {
      value = dense<0.0625> : tensor<f32>
    } : () -> tensor<f32>
    %relu_rhs_zp = "onnx.Constant"() {
      value = dense<5> : tensor<ui8>
    } : () -> tensor<ui8>
    %relu_rhs = "onnx.DequantizeLinear"(
        %relu_rhs_q, %relu_rhs_scale, %relu_rhs_zp) {axis = 1 : si64}
        : (tensor<1x8x2x4xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xf32>
    %relu_add = "onnx.Add"(%relu_lhs, %relu_rhs)
        : (tensor<1x8x2x4xf32>, tensor<1x8x2x4xf32>)
          -> tensor<1x8x2x4xf32>
    %relu = "onnx.Relu"(%relu_add)
        : (tensor<1x8x2x4xf32>) -> tensor<1x8x2x4xf32>
    %relu_output_scale = "onnx.Constant"() {
      value = dense<0.25> : tensor<f32>
    } : () -> tensor<f32>
    %relu_output_zp = "onnx.Constant"() {
      value = dense<7> : tensor<ui8>
    } : () -> tensor<ui8>
    %relu_output = "onnx.QuantizeLinear"(
        %relu, %relu_output_scale, %relu_output_zp) {axis = 1 : si64}
        : (tensor<1x8x2x4xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xui8>

    %direct_lhs_scale = "onnx.Constant"() {
      value = dense<0.5> : tensor<f32>
    } : () -> tensor<f32>
    %direct_lhs_zp = "onnx.Constant"() {
      value = dense<11> : tensor<ui8>
    } : () -> tensor<ui8>
    %direct_lhs = "onnx.DequantizeLinear"(
        %direct_lhs_q, %direct_lhs_scale, %direct_lhs_zp) {axis = 1 : si64}
        : (tensor<1x8x2x4xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xf32>
    %direct_rhs_scale = "onnx.Constant"() {
      value = dense<0.25> : tensor<f32>
    } : () -> tensor<f32>
    %direct_rhs_zp = "onnx.Constant"() {
      value = dense<13> : tensor<ui8>
    } : () -> tensor<ui8>
    %direct_rhs = "onnx.DequantizeLinear"(
        %direct_rhs_q, %direct_rhs_scale, %direct_rhs_zp) {axis = 1 : si64}
        : (tensor<1x8x2x4xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xf32>
    %direct_add = "onnx.Add"(%direct_lhs, %direct_rhs)
        : (tensor<1x8x2x4xf32>, tensor<1x8x2x4xf32>)
          -> tensor<1x8x2x4xf32>
    %direct_output_scale = "onnx.Constant"() {
      value = dense<0.25> : tensor<f32>
    } : () -> tensor<f32>
    %direct_output_zp = "onnx.Constant"() {
      value = dense<17> : tensor<ui8>
    } : () -> tensor<ui8>
    %direct_output = "onnx.QuantizeLinear"(
        %direct_add, %direct_output_scale, %direct_output_zp) {axis = 1 : si64}
        : (tensor<1x8x2x4xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xui8>

    return %relu_output, %direct_output
        : tensor<1x8x2x4xui8>, tensor<1x8x2x4xui8>
  }
}

// CHECK-LABEL: func.func @lower_qadd
// CHECK: %[[CONV_Q:.*]] = standalone.conv_requant
// CHECK-SAME: -> tensor<1x8x2x4xui8>
// CHECK: %[[RELU_QADD:.*]] = standalone.qadd_relu
// CHECK-SAME: ins(%[[CONV_Q]],
// CHECK-SAME: lhs_multiplier = 1073741824 : i32
// CHECK-SAME: lhs_zero_point = 3 : i32
// CHECK-SAME: output_zero_point = 7 : i32
// CHECK-SAME: relu_enable = 1 : i32
// CHECK-SAME: rhs_multiplier = 536870912 : i32
// CHECK-SAME: rhs_zero_point = 5 : i32
// CHECK-SAME: shift = 31 : i32
// CHECK: %[[DIRECT_QADD:.*]] = standalone.qadd_relu
// CHECK-SAME: lhs_multiplier = 1073741824 : i32
// CHECK-SAME: lhs_zero_point = 11 : i32
// CHECK-SAME: output_zero_point = 17 : i32
// CHECK-SAME: relu_enable = 0 : i32
// CHECK-SAME: rhs_multiplier = 536870912 : i32
// CHECK-SAME: rhs_zero_point = 13 : i32
// CHECK-SAME: shift = 29 : i32
// CHECK-NOT: "onnx.DequantizeLinear"
// CHECK-NOT: "onnx.Add"
// CHECK-NOT: "onnx.Relu"
// CHECK-NOT: "onnx.QuantizeLinear"
// CHECK: return %[[RELU_QADD]], %[[DIRECT_QADD]]
