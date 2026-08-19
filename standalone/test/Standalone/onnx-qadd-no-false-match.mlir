// RUN: standalone-opt --allow-unregistered-dialect \
// RUN:   --transform-preload-library="transform-library-paths=%S/../../../projects/resnet18/qadd_relu.mlir" \
// RUN:   --transform-interpreter %s | FileCheck %s

module {
  func.func @relu_without_add(%input: tensor<1x8x2x4xf32>)
      -> tensor<1x8x2x4xui8> {
    %relu = "onnx.Relu"(%input)
        : (tensor<1x8x2x4xf32>) -> tensor<1x8x2x4xf32>
    %scale = "onnx.Constant"() {value = dense<0.25> : tensor<f32>}
        : () -> tensor<f32>
    %zp = "onnx.Constant"() {value = dense<7> : tensor<ui8>}
        : () -> tensor<ui8>
    %output = "onnx.QuantizeLinear"(%relu, %scale, %zp)
        : (tensor<1x8x2x4xf32>, tensor<f32>, tensor<ui8>)
          -> tensor<1x8x2x4xui8>
    return %output : tensor<1x8x2x4xui8>
  }
}

// CHECK-LABEL: func.func @relu_without_add
// CHECK: "onnx.Relu"
// CHECK: "onnx.QuantizeLinear"
// CHECK-NOT: standalone.qadd_relu
