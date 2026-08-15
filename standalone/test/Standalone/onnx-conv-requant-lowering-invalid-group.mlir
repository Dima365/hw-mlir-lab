// RUN: not standalone-opt --allow-unregistered-dialect \
// RUN:   --transform-interpreter %s 2>&1 | FileCheck %s

module attributes {transform.with_named_sequence} {
  func.func @reject_grouped_conv(%input_q: tensor<1x2x2x2xui8>) {
    %activation_scale = "onnx.Constant"() {
      value = dense<0.5> : tensor<f32>
    } : () -> tensor<f32>
    %activation_zero_point = "onnx.Constant"() {
      value = dense<3> : tensor<ui8>
    } : () -> tensor<ui8>
    %input = "onnx.DequantizeLinear"(
        %input_q, %activation_scale, %activation_zero_point)
        : (tensor<1x2x2x2xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x2x2x2xf32>

    %weight_q = "onnx.Constant"() {
      value = dense<0> : tensor<2x1x1x1xi8>
    } : () -> tensor<2x1x1x1xi8>
    %weight_scale = "onnx.Constant"() {
      value = dense<0.25> : tensor<2xf32>
    } : () -> tensor<2xf32>
    %weight_zero_point = "onnx.Constant"() {
      value = dense<0> : tensor<2xi8>
    } : () -> tensor<2xi8>
    %weight = "onnx.DequantizeLinear"(
        %weight_q, %weight_scale, %weight_zero_point) {axis = 0 : si64}
        : (tensor<2x1x1x1xi8>, tensor<2xf32>, tensor<2xi8>)
          -> tensor<2x1x1x1xf32>

    %bias_q = "onnx.Constant"() {
      value = dense<0> : tensor<2xi32>
    } : () -> tensor<2xi32>
    %bias_scale = "onnx.Constant"() {
      value = dense<0.125> : tensor<2xf32>
    } : () -> tensor<2xf32>
    %bias_zero_point = "onnx.Constant"() {
      value = dense<0> : tensor<2xi32>
    } : () -> tensor<2xi32>
    %bias = "onnx.DequantizeLinear"(
        %bias_q, %bias_scale, %bias_zero_point) {axis = 0 : si64}
        : (tensor<2xi32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2xf32>

    %output_scale = "onnx.Constant"() {
      value = dense<0.25> : tensor<f32>
    } : () -> tensor<f32>
    %output_zero_point = "onnx.Constant"() {
      value = dense<7> : tensor<ui8>
    } : () -> tensor<ui8>
    %conv = "onnx.Conv"(%input, %weight, %bias) {
      group = 2 : si64, kernel_shape = [1, 1], strides = [1, 1],
      pads = [0, 0, 0, 0], dilations = [1, 1]
    } : (tensor<1x2x2x2xf32>, tensor<2x1x1x1xf32>, tensor<2xf32>)
      -> tensor<1x2x2x2xf32>
    %quant = "onnx.QuantizeLinear"(
        %conv, %output_scale, %output_zero_point)
        : (tensor<1x2x2x2xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x2x2x2xui8>
    return
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %conv = transform.structured.match ops{["onnx.Conv"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %quant = transform.structured.match ops{["onnx.QuantizeLinear"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %multipliers = transform.param.constant array<i64: 1, 1>
        -> !transform.any_param
    %shifts = transform.param.constant array<i64: 1, 1>
        -> !transform.any_param
    %output_zero_point = transform.param.constant 7 : i64
        -> !transform.any_param
    %accumulate, %requant = transform.standalone.lower_onnx_conv_requant
        %conv, %quant, %multipliers, %shifts, %output_zero_point
        : (!transform.any_op, !transform.any_op,
           !transform.any_param, !transform.any_param,
           !transform.any_param) ->
          (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: error: requantization at index 0: onnx.Conv group must be 1
