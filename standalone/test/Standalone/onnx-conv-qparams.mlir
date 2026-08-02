// RUN: standalone-opt --allow-unregistered-dialect \
// RUN:   --transform-preload-library="transform-library-paths=%S/Inputs/onnx-conv-qparams-report.mlir" \
// RUN:   --transform-interpreter %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @extract_qparams(%input_q: tensor<1x2x4x4xui8>) {
    %activation_scale = "onnx.Constant"() {value = dense<0.5> : tensor<f32>}
        : () -> tensor<f32>
    %activation_zp = "onnx.Constant"() {value = dense<3> : tensor<ui8>}
        : () -> tensor<ui8>
    %input = "onnx.DequantizeLinear"(
        %input_q, %activation_scale, %activation_zp) {axis = 1 : si64}
        : (tensor<1x2x4x4xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x2x4x4xf32>

    %weight_q = "onnx.Constant"() {value = dense<0> : tensor<2x2x1x1xi8>}
        : () -> tensor<2x2x1x1xi8>
    %weight_scale = "onnx.Constant"() {
        value = dense<[0.25, 0.125]> : tensor<2xf32>
      } : () -> tensor<2xf32>
    %weight_zp = "onnx.Constant"() {value = dense<0> : tensor<2xi8>}
        : () -> tensor<2xi8>
    %weight = "onnx.DequantizeLinear"(
        %weight_q, %weight_scale, %weight_zp) {axis = 0 : si64}
        : (tensor<2x2x1x1xi8>, tensor<2xf32>, tensor<2xi8>)
          -> tensor<2x2x1x1xf32>

    %bias_q = "onnx.Constant"() {value = dense<[4, -2]> : tensor<2xi32>}
        : () -> tensor<2xi32>
    %bias_scale = "onnx.Constant"() {
        value = dense<[0.125, 0.0625]> : tensor<2xf32>
      } : () -> tensor<2xf32>
    %bias_zp = "onnx.Constant"() {value = dense<0> : tensor<2xi32>}
        : () -> tensor<2xi32>
    %bias = "onnx.DequantizeLinear"(
        %bias_q, %bias_scale, %bias_zp) {axis = 0 : si64}
        : (tensor<2xi32>, tensor<2xf32>, tensor<2xi32>) -> tensor<2xf32>

    %conv = "onnx.Conv"(%input, %weight, %bias)
        : (tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>, tensor<2xf32>)
          -> tensor<1x2x4x4xf32>
    %relu = "onnx.Relu"(%conv)
        : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %output_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>}
        : () -> tensor<f32>
    %output_zp = "onnx.Constant"() {value = dense<7> : tensor<ui8>}
        : () -> tensor<ui8>
    %output = "onnx.QuantizeLinear"(%relu, %output_scale, %output_zp)
        {axis = 1 : si64, saturate = 1 : si64}
        : (tensor<1x2x4x4xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x2x4x4xui8>
    return
  }
}

// CHECK: activation_scale 5.000000e-01 : f64
// CHECK: activation_zero_point 3 : i64
// CHECK: weight_scales array<f64: 2.500000e-01, 1.250000e-01>
// CHECK: weight_zero_points array<i64: 0, 0>
// CHECK: output_scale 2.500000e-01 : f64
// CHECK: output_zero_point 7 : i64
// CHECK: bias_scales array<f64: 1.250000e-01, 6.250000e-02>
// CHECK: bias_zero_points array<i64: 0, 0>
