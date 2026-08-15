// RUN: standalone-opt --allow-unregistered-dialect --transform-interpreter %s \
// RUN:   | FileCheck %s

module attributes {transform.with_named_sequence} {
  func.func @lower_conv_requant(%input_q: tensor<1x1x2x4xui8>)
      -> (tensor<1x8x2x4xui8>, tensor<1x8x2x4xui8>) {
    %activation_scale = "onnx.Constant"() {
      value = dense<0.5> : tensor<f32>
    } : () -> tensor<f32>
    %activation_zero_point = "onnx.Constant"() {
      value = dense<3> : tensor<ui8>
    } : () -> tensor<ui8>
    %input = "onnx.DequantizeLinear"(
        %input_q, %activation_scale, %activation_zero_point)
        : (tensor<1x1x2x4xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x1x2x4xf32>

    %weight_q = "onnx.Constant"() {
      value = dense<0> : tensor<8x1x1x1xi8>
    } : () -> tensor<8x1x1x1xi8>
    %weight_scale = "onnx.Constant"() {
      value = dense<[0.25, 0.125, 0.0625, 0.03125,
                     0.25, 0.125, 0.0625, 0.03125]> : tensor<8xf32>
    } : () -> tensor<8xf32>
    %weight_zero_point = "onnx.Constant"() {
      value = dense<0> : tensor<8xi8>
    } : () -> tensor<8xi8>
    %weight = "onnx.DequantizeLinear"(
        %weight_q, %weight_scale, %weight_zero_point) {axis = 0 : si64}
        : (tensor<8x1x1x1xi8>, tensor<8xf32>, tensor<8xi8>)
          -> tensor<8x1x1x1xf32>

    %bias_q = "onnx.Constant"() {
      value = dense<0> : tensor<8xi32>
    } : () -> tensor<8xi32>
    %bias_scale = "onnx.Constant"() {
      value = dense<[0.125, 0.0625, 0.03125, 0.015625,
                     0.125, 0.0625, 0.03125, 0.015625]> : tensor<8xf32>
    } : () -> tensor<8xf32>
    %bias_zero_point = "onnx.Constant"() {
      value = dense<0> : tensor<8xi32>
    } : () -> tensor<8xi32>
    %bias = "onnx.DequantizeLinear"(
        %bias_q, %bias_scale, %bias_zero_point) {axis = 0 : si64}
        : (tensor<8xi32>, tensor<8xf32>, tensor<8xi32>) -> tensor<8xf32>

    %output_scale = "onnx.Constant"() {
      value = dense<0.25> : tensor<f32>
    } : () -> tensor<f32>
    %output_zero_point = "onnx.Constant"() {
      value = dense<7> : tensor<ui8>
    } : () -> tensor<ui8>

    %conv_relu = "onnx.Conv"(%input, %weight, %bias) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]
    }
        : (tensor<1x1x2x4xf32>, tensor<8x1x1x1xf32>, tensor<8xf32>)
          -> tensor<1x8x2x4xf32>
    %relu = "onnx.Relu"(%conv_relu)
        : (tensor<1x8x2x4xf32>) -> tensor<1x8x2x4xf32>
    %quant_relu = "onnx.QuantizeLinear"(
        %relu, %output_scale, %output_zero_point)
        : (tensor<1x8x2x4xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xui8>

    %conv_direct = "onnx.Conv"(%input, %weight, %bias) {
      auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64,
      kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]
    }
        : (tensor<1x1x2x4xf32>, tensor<8x1x1x1xf32>, tensor<8xf32>)
          -> tensor<1x8x2x4xf32>
    %quant_direct = "onnx.QuantizeLinear"(
        %conv_direct, %output_scale, %output_zero_point)
        : (tensor<1x8x2x4xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xui8>

    return %quant_relu, %quant_direct
        : tensor<1x8x2x4xui8>, tensor<1x8x2x4xui8>
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %convs = transform.structured.match ops{["onnx.Conv"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %conv_relu, %conv_direct = transform.split_handle %convs
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %relus = transform.structured.match ops{["onnx.Relu"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %quants = transform.structured.match ops{["onnx.QuantizeLinear"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %quant_relu, %quant_direct = transform.split_handle %quants
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %relu_activation_scale, %relu_activation_zero_point,
    %relu_weight_scales, %relu_weight_zero_points,
    %relu_output_scale, %relu_output_zero_point,
    %relu_bias_scales, %relu_bias_zero_points =
        transform.standalone.extract_onnx_conv_qparams %conv_relu
        : (!transform.any_op) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param)

    %direct_activation_scale, %direct_activation_zero_point,
    %direct_weight_scales, %direct_weight_zero_points,
    %direct_output_scale, %direct_output_zero_point,
    %direct_bias_scales, %direct_bias_zero_points =
        transform.standalone.extract_onnx_conv_qparams %conv_direct
        : (!transform.any_op) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param)

    %relu_multipliers, %relu_shifts =
        transform.standalone.compute_conv_requant_fixed_point
            %relu_activation_scale, %relu_weight_scales,
            %relu_weight_zero_points, %relu_output_scale
        : (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param) ->
          (!transform.any_param, !transform.any_param)

    %direct_multipliers, %direct_shifts =
        transform.standalone.compute_conv_requant_fixed_point
            %direct_activation_scale, %direct_weight_scales,
            %direct_weight_zero_points, %direct_output_scale
        : (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param) ->
          (!transform.any_param, !transform.any_param)

    %relu_accumulate, %relu_requant =
        transform.standalone.lower_onnx_conv_relu_requant
            %conv_relu, %relus, %quant_relu,
            %relu_multipliers, %relu_shifts, %relu_output_zero_point
        : (!transform.any_op, !transform.any_op, !transform.any_op,
           !transform.any_param, !transform.any_param,
           !transform.any_param) ->
          (!transform.any_op, !transform.any_op)

    %direct_accumulate, %direct_requant =
        transform.standalone.lower_onnx_conv_requant
            %conv_direct, %quant_direct, %direct_multipliers,
            %direct_shifts, %direct_output_zero_point
        : (!transform.any_op, !transform.any_op,
           !transform.any_param, !transform.any_param,
           !transform.any_param) ->
          (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @lower_conv_requant
// CHECK: standalone.conv_accumulate
// CHECK-SAME: input_zero_point = 3 : i32
// CHECK-SAME: -> tensor<1x8x2x4xi32>
// CHECK: arith.constant dense<[1073741824, 536870912, 268435456, 134217728, 1073741824, 536870912, 268435456, 134217728]> : tensor<8xi32>
// CHECK-NEXT: arith.constant dense<31> : tensor<8xi32>
// CHECK: standalone.conv_requant
// CHECK-SAME: output_zero_point = 7 : i32
// CHECK-SAME: relu_enable = 1 : i32
// CHECK-SAME: -> tensor<1x8x2x4xui8>
// CHECK: standalone.conv_accumulate
// CHECK-SAME: input_zero_point = 3 : i32
// CHECK-SAME: -> tensor<1x8x2x4xi32>
// CHECK: arith.constant dense<[1073741824, 536870912, 268435456, 134217728, 1073741824, 536870912, 268435456, 134217728]> : tensor<8xi32>
// CHECK-NEXT: arith.constant dense<31> : tensor<8xi32>
// CHECK: standalone.conv_requant
// CHECK-SAME: output_zero_point = 7 : i32
// CHECK-SAME: relu_enable = 0 : i32
// CHECK-SAME: -> tensor<1x8x2x4xui8>
// CHECK-NOT: "onnx.Conv"
// CHECK-NOT: "onnx.Relu"
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return {{.*}} : tensor<1x8x2x4xui8>, tensor<1x8x2x4xui8>
