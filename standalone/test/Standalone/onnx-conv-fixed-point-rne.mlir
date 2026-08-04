// RUN: standalone-opt --allow-unregistered-dialect \
// RUN:   --transform-preload-library="transform-library-paths=%S/Inputs/onnx-conv-qparams-report.mlir" \
// RUN:   --transform-interpreter %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @fixed_point_rne(%input_q: tensor<1x1x2x2xui8>) {
    %as = "onnx.Constant"() {
        value = dense<0.5000000596046448> : tensor<f32>
      } : () -> tensor<f32>
    %az = "onnx.Constant"() {value = dense<0> : tensor<ui8>}
        : () -> tensor<ui8>
    %input = "onnx.DequantizeLinear"(%input_q, %as, %az) {axis = 1 : si64}
        : (tensor<1x1x2x2xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x1x2x2xf32>

    %wq = "onnx.Constant"() {value = dense<0> : tensor<3x1x1x1xi8>}
        : () -> tensor<3x1x1x1xi8>
    %ws = "onnx.Constant"() {
        value = dense<[0.50390625, 0.51171875, 4.0]> : tensor<3xf32>
      } : () -> tensor<3xf32>
    %wz = "onnx.Constant"() {value = dense<0> : tensor<3xi8>}
        : () -> tensor<3xi8>
    %weight = "onnx.DequantizeLinear"(%wq, %ws, %wz) {axis = 0 : si64}
        : (tensor<3x1x1x1xi8>, tensor<3xf32>, tensor<3xi8>)
          -> tensor<3x1x1x1xf32>

    %bq = "onnx.Constant"() {value = dense<0> : tensor<3xi32>}
        : () -> tensor<3xi32>
    %bs = "onnx.Constant"() {
        value = dense<[0.2519531548023224,
                       0.2558594048023224,
                       2.000000238418579]> : tensor<3xf32>
      } : () -> tensor<3xf32>
    %bz = "onnx.Constant"() {value = dense<0> : tensor<3xi32>}
        : () -> tensor<3xi32>
    %bias = "onnx.DequantizeLinear"(%bq, %bs, %bz) {axis = 0 : si64}
        : (tensor<3xi32>, tensor<3xf32>, tensor<3xi32>) -> tensor<3xf32>

    %conv = "onnx.Conv"(%input, %weight, %bias)
        : (tensor<1x1x2x2xf32>, tensor<3x1x1x1xf32>, tensor<3xf32>)
          -> tensor<1x3x2x2xf32>
    %os = "onnx.Constant"() {value = dense<1.0> : tensor<f32>}
        : () -> tensor<f32>
    %oz = "onnx.Constant"() {value = dense<0> : tensor<ui8>}
        : () -> tensor<ui8>
    %out = "onnx.QuantizeLinear"(%conv, %os, %oz)
        : (tensor<1x3x2x2xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x3x2x2xui8>
    return
  }
}

// Both scaled values end in exactly .5. RNE keeps 541065280 and increments
// the odd 549453889 to the next even integer.
// The third coefficient selects shift 29 because larger shifts overflow i32.
// CHECK: multipliers array<i64: 541065280, 549453890, 1073741952>
// CHECK: shifts array<i64: 31, 31, 29>
