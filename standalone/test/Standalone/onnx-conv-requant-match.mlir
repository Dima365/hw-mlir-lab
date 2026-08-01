// RUN: standalone-opt --allow-unregistered-dialect \
// RUN:   --transform-preload-library="transform-library-paths=%S/../../../projects/resnet18/conv_requant_match.mlir" \
// RUN:   --transform-interpreter %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s \
// RUN:       --implicit-check-not='remark: loc("non_conv_quantize")' \
// RUN:       --implicit-check-not='remark: loc("non_conv_relu_quantize")'

module {
  func.func @match_conv_requant(
      %input: tensor<1x3x8x8xf32>,
      %weight: tensor<8x3x3x3xf32>,
      %bias: tensor<8xf32>,
      %scale: tensor<f32>,
      %zero_point: tensor<ui8>) {
    %conv_with_relu = "onnx.Conv"(%input, %weight, %bias)
        : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>)
          -> tensor<1x8x8x8xf32>
    %relu = "onnx.Relu"(%conv_with_relu)
        : (tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
    %quant_after_relu =
        "onnx.QuantizeLinear"(%relu, %scale, %zero_point)
        : (tensor<1x8x8x8xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x8x8xui8>
        loc("conv_relu_quantize")

    %conv_without_relu = "onnx.Conv"(%input, %weight, %bias)
        : (tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>)
          -> tensor<1x8x8x8xf32>
    %quant_after_conv =
        "onnx.QuantizeLinear"(%conv_without_relu, %scale, %zero_point)
        : (tensor<1x8x8x8xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x8x8xui8>
        loc("conv_quantize")

    %add = "onnx.Add"(%input, %input)
        : (tensor<1x3x8x8xf32>, tensor<1x3x8x8xf32>)
          -> tensor<1x3x8x8xf32>
    %quant_after_add = "onnx.QuantizeLinear"(%add, %scale, %zero_point)
        : (tensor<1x3x8x8xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x3x8x8xui8>
        loc("non_conv_quantize")

    %relu_without_conv = "onnx.Relu"(%input)
        : (tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32>
    %quant_after_non_conv_relu =
        "onnx.QuantizeLinear"(%relu_without_conv, %scale, %zero_point)
        : (tensor<1x3x8x8xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x3x8x8xui8>
        loc("non_conv_relu_quantize")

    return
  }
}

// CHECK-COUNT-1: remark: loc("conv_relu_quantize"): matched Conv -> Relu -> QuantizeLinear
// CHECK-COUNT-1: remark: loc("conv_quantize"): matched Conv -> QuantizeLinear
