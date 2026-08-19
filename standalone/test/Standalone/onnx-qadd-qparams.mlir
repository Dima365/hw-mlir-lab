// RUN: standalone-opt --allow-unregistered-dialect \
// RUN:   --transform-preload-library="transform-library-paths=%S/Inputs/onnx-qadd-qparams-report.mlir" \
// RUN:   --transform-interpreter %s -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @extract_qadd_params(
      %lhs_q: tensor<1x8x2x4xui8>,
      %rhs_q: tensor<1x8x2x4xui8>) -> tensor<1x8x2x4xui8> {
    %lhs_scale = "onnx.Constant"() {value = dense<0.125> : tensor<f32>}
        : () -> tensor<f32>
    %lhs_zp = "onnx.Constant"() {value = dense<3> : tensor<ui8>}
        : () -> tensor<ui8>
    %lhs = "onnx.DequantizeLinear"(%lhs_q, %lhs_scale, %lhs_zp)
        {axis = 1 : si64}
        : (tensor<1x8x2x4xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xf32>

    %rhs_scale = "onnx.Constant"() {value = dense<0.0625> : tensor<f32>}
        : () -> tensor<f32>
    %rhs_zp = "onnx.Constant"() {value = dense<5> : tensor<ui8>}
        : () -> tensor<ui8>
    %rhs = "onnx.DequantizeLinear"(%rhs_q, %rhs_scale, %rhs_zp)
        {axis = 1 : si64}
        : (tensor<1x8x2x4xui8>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xf32>

    %add = "onnx.Add"(%lhs, %rhs)
        : (tensor<1x8x2x4xf32>, tensor<1x8x2x4xf32>)
          -> tensor<1x8x2x4xf32>
    %relu = "onnx.Relu"(%add)
        : (tensor<1x8x2x4xf32>) -> tensor<1x8x2x4xf32>
    %output_scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>}
        : () -> tensor<f32>
    %output_zp = "onnx.Constant"() {value = dense<7> : tensor<ui8>}
        : () -> tensor<ui8>
    %output = "onnx.QuantizeLinear"(%relu, %output_scale, %output_zp)
        {axis = 1 : si64}
        : (tensor<1x8x2x4xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xui8>
    return %output : tensor<1x8x2x4xui8>
  }
}

// CHECK: lhs_scale 1.250000e-01 : f64
// CHECK: lhs_zero_point 3 : i64
// CHECK: rhs_scale 6.250000e-02 : f64
// CHECK: rhs_zero_point 5 : i64
// CHECK: output_scale 2.500000e-01 : f64
// CHECK: output_zero_point 7 : i64
// CHECK: lhs_multiplier 1073741824 : i64
// CHECK: rhs_multiplier 536870912 : i64
// CHECK: shift 31 : i64
