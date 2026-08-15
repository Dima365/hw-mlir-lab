// RUN: standalone-opt --allow-unregistered-dialect \
// RUN:   --transform-preload-library="transform-library-paths=%S/../../../projects/resnet18/conv_requant_match.mlir" \
// RUN:   --transform-interpreter %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s \
// RUN:       --implicit-check-not='remark: loc("non_conv_quantize")' \
// RUN:       --implicit-check-not='remark: loc("non_conv_relu_quantize")'

module {
  func.func @match_conv_requant(
      %input_q: tensor<1x3x8x8xui8>) {
    %activation_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>}
        : () -> tensor<f32>
    %activation_zero_point = "onnx.Constant"() {value = dense<3> : tensor<ui8>}
        : () -> tensor<ui8>
    %input = "onnx.DequantizeLinear"(
        %input_q, %activation_scale, %activation_zero_point)
        {axis = 1 : si64}
        : (tensor<1x3x8x8xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x3x8x8xf32>

    %weight_q = "onnx.Constant"() {value = dense<0> : tensor<8x3x3x3xi8>}
        : () -> tensor<8x3x3x3xi8>
    %weight_scale = "onnx.Constant"() {
        value = dense<[0.25, 0.125, 0.0625, 0.03125,
                       0.25, 0.125, 0.0625, 0.03125]> : tensor<8xf32>
      } : () -> tensor<8xf32>
    %weight_zero_point = "onnx.Constant"() {value = dense<0> : tensor<8xi8>}
        : () -> tensor<8xi8>
    %weight = "onnx.DequantizeLinear"(
        %weight_q, %weight_scale, %weight_zero_point)
        {axis = 0 : si64}
        : (tensor<8x3x3x3xi8>, tensor<8xf32>, tensor<8xi8>)
          -> tensor<8x3x3x3xf32>

    %bias_q = "onnx.Constant"() {value = dense<0> : tensor<8xi32>}
        : () -> tensor<8xi32>
    %bias_scale = "onnx.Constant"() {
        value = dense<[0.125, 0.0625, 0.03125, 0.015625,
                       0.125, 0.0625, 0.03125, 0.015625]> : tensor<8xf32>
      } : () -> tensor<8xf32>
    %bias_zero_point = "onnx.Constant"() {value = dense<0> : tensor<8xi32>}
        : () -> tensor<8xi32>
    %bias = "onnx.DequantizeLinear"(
        %bias_q, %bias_scale, %bias_zero_point)
        {axis = 0 : si64}
        : (tensor<8xi32>, tensor<8xf32>, tensor<8xi32>) -> tensor<8xf32>

    %output_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>}
        : () -> tensor<f32>
    %output_zero_point = "onnx.Constant"() {value = dense<7> : tensor<ui8>}
        : () -> tensor<ui8>

    %conv_with_relu = "onnx.Conv"(%input, %weight, %bias) {
      dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
      pads = [1, 1, 1, 1], strides = [1, 1]
    }
        : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>)
          -> tensor<1x8x8x8xf32>
    %relu = "onnx.Relu"(%conv_with_relu)
        : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
    %quant_after_relu =
        "onnx.QuantizeLinear"(%relu, %output_scale, %output_zero_point)
        : (tensor<1x8x8x8xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x8x8xui8>
        loc("conv_relu_quantize")

    %conv_without_relu = "onnx.Conv"(%input, %weight, %bias) {
      dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
      pads = [1, 1, 1, 1], strides = [1, 1]
    }
        : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>)
          -> tensor<1x8x8x8xf32>
    %quant_after_conv =
        "onnx.QuantizeLinear"(
            %conv_without_relu, %output_scale, %output_zero_point)
        : (tensor<1x8x8x8xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x8x8xui8>
        loc("conv_quantize")

    %add = "onnx.Add"(%input, %input)
        : (tensor<1x3x8x8xf32>, tensor<1x3x8x8xf32>)
          -> tensor<1x3x8x8xf32>
    %quant_after_add = "onnx.QuantizeLinear"(
        %add, %output_scale, %output_zero_point)
        : (tensor<1x3x8x8xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x3x8x8xui8>
        loc("non_conv_quantize")

    %relu_without_conv = "onnx.Relu"(%input)
        : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
    %quant_after_non_conv_relu = "onnx.QuantizeLinear"(
        %relu_without_conv, %output_scale, %output_zero_point)
        : (tensor<1x3x8x8xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x3x8x8xui8>
        loc("non_conv_relu_quantize")

    return
  }
}

// CHECK-COUNT-1: remark: loc("conv_relu_quantize"): matched Conv -> Relu -> QuantizeLinear
// CHECK-COUNT-1: remark: loc("conv_quantize"): matched Conv -> QuantizeLinear
