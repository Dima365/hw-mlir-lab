// RUN: not standalone-opt --allow-unregistered-dialect \
// RUN:   --transform-preload-library="transform-library-paths=%S/Inputs/onnx-qadd-qparams-report.mlir" \
// RUN:   --transform-interpreter %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @reject_per_channel_scale(
      %lhs_q: tensor<1x8x2x4xui8>,
      %rhs_q: tensor<1x8x2x4xui8>) -> tensor<1x8x2x4xui8> {
    %lhs_scale = "onnx.Constant"() {value = dense<0.125> : tensor<8xf32>}
        : () -> tensor<8xf32>
    %lhs_zp = "onnx.Constant"() {value = dense<3> : tensor<ui8>}
        : () -> tensor<ui8>
    %lhs = "onnx.DequantizeLinear"(%lhs_q, %lhs_scale, %lhs_zp)
        {axis = 1 : si64}
        : (tensor<1x8x2x4xui8>, tensor<8xf32>, tensor<ui8>)
          -> tensor<1x8x2x4xf32>

    %rhs_scale = "onnx.Constant"() {value = dense<0.0625> : tensor<f32>}
        : () -> tensor<f32>
    %rhs_zp = "onnx.Constant"() {value = dense<5> : tensor<ui8>}
        : () -> tensor<ui8>
    %rhs = "onnx.DequantizeLinear"(%rhs_q, %rhs_scale, %rhs_zp)
        : (tensor<1x8x2x4xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xf32>

    %add = "onnx.Add"(%lhs, %rhs)
        : (tensor<1x8x2x4xf32>, tensor<1x8x2x4xf32>)
          -> tensor<1x8x2x4xf32>
    %output_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>}
        : () -> tensor<f32>
    %output_zp = "onnx.Constant"() {value = dense<7> : tensor<ui8>}
        : () -> tensor<ui8>
    %output = "onnx.QuantizeLinear"(%add, %output_scale, %output_zp)
        : (tensor<1x8x2x4xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xui8>
    return %output : tensor<1x8x2x4xui8>
  }
}

// CHECK: lhs scale must be tensor<f32>
