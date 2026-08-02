// RUN: not standalone-opt --allow-unregistered-dialect \
// RUN:   --transform-preload-library="transform-library-paths=%S/Inputs/onnx-conv-qparams-report.mlir" \
// RUN:   --transform-interpreter %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @bad_bias_scale(%input_q: tensor<1x1x2x2xui8>) {
    %as = "onnx.Constant"() {value = dense<0.5> : tensor<f32>}
        : () -> tensor<f32>
    %az = "onnx.Constant"() {value = dense<0> : tensor<ui8>}
        : () -> tensor<ui8>
    %input = "onnx.DequantizeLinear"(%input_q, %as, %az) {axis = 1 : si64}
        : (tensor<1x1x2x2xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x1x2x2xf32>

    %wq = "onnx.Constant"() {value = dense<0> : tensor<1x1x1x1xi8>}
        : () -> tensor<1x1x1x1xi8>
    %ws = "onnx.Constant"() {value = dense<0.25> : tensor<1xf32>}
        : () -> tensor<1xf32>
    %wz = "onnx.Constant"() {value = dense<0> : tensor<1xi8>}
        : () -> tensor<1xi8>
    %weight = "onnx.DequantizeLinear"(%wq, %ws, %wz) {axis = 0 : si64}
        : (tensor<1x1x1x1xi8>, tensor<1xf32>, tensor<1xi8>)
          -> tensor<1x1x1x1xf32>

    %bq = "onnx.Constant"() {value = dense<0> : tensor<1xi32>}
        : () -> tensor<1xi32>
    %bs = "onnx.Constant"() {value = dense<0.25> : tensor<1xf32>}
        : () -> tensor<1xf32>
    %bz = "onnx.Constant"() {value = dense<0> : tensor<1xi32>}
        : () -> tensor<1xi32>
    %bias = "onnx.DequantizeLinear"(%bq, %bs, %bz) {axis = 0 : si64}
        : (tensor<1xi32>, tensor<1xf32>, tensor<1xi32>) -> tensor<1xf32>

    %conv = "onnx.Conv"(%input, %weight, %bias)
        : (tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>)
          -> tensor<1x1x2x2xf32>
    %os = "onnx.Constant"() {value = dense<0.25> : tensor<f32>}
        : () -> tensor<f32>
    %oz = "onnx.Constant"() {value = dense<0> : tensor<ui8>}
        : () -> tensor<ui8>
    %out = "onnx.QuantizeLinear"(%conv, %os, %oz)
        : (tensor<1x1x2x2xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x1x2x2xui8>
    return
  }
}

// CHECK: bias scale at channel 0 does not match activation_scale * weight_scale
