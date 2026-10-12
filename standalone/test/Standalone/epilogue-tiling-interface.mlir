// RUN: standalone-opt --transform-interpreter %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  func.func @matrix(%acc: tensor<16x24xi32>, %mult: tensor<24xi32>,
                    %shift: tensor<24xi32>, %rhs: tensor<16x24xui8>)
      -> tensor<16x24xui8> {
    %empty = tensor.empty() : tensor<16x24xui8>
    %q = standalone.conv_requant
      ins(%acc, %mult, %shift : tensor<16x24xi32>, tensor<24xi32>, tensor<24xi32>)
      outs(%empty : tensor<16x24xui8>)
      {output_zero_point = 37 : i32, relu_enable = 0 : i32}
      -> tensor<16x24xui8>
    %sum = standalone.qadd_relu
      ins(%q, %rhs : tensor<16x24xui8>, tensor<16x24xui8>)
      outs(%empty : tensor<16x24xui8>)
      {lhs_multiplier = 3 : i32, rhs_multiplier = 5 : i32, shift = 4 : i32,
       lhs_zero_point = 37 : i32, rhs_zero_point = 113 : i32,
       output_zero_point = 29 : i32, relu_enable = 1 : i32}
      -> tensor<16x24xui8>
    return %sum : tensor<16x24xui8>
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %q = transform.structured.match ops{["standalone.conv_requant"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %m, %c = transform.structured.tile_using_for %q tile_sizes [8, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %add = transform.structured.match ops{["standalone.qadd_relu"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled_add, %am, %ac = transform.structured.tile_using_for %add tile_sizes [8, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: func.func @matrix
// CHECK: scf.for %[[M:[a-zA-Z0-9_]+]] =
// CHECK: scf.for %[[C:[a-zA-Z0-9_]+]] =
// CHECK: tensor.extract_slice %arg0[%[[M]], %[[C]]] [8, 8] [1, 1]
// CHECK: tensor.extract_slice %arg1[%[[C]]] [8] [1]
// CHECK: tensor.extract_slice %arg2[%[[C]]] [8] [1]
// CHECK: standalone.conv_requant
// CHECK-SAME: tensor<8x8xi32>, tensor<8xi32>, tensor<8xi32>
// CHECK-SAME: output_zero_point = 37 : i32, relu_enable = 0 : i32
// CHECK: standalone.qadd_relu
// CHECK-SAME: tensor<8x8xui8>, tensor<8x8xui8>
// CHECK-SAME: lhs_multiplier = 3 : i32
// CHECK-SAME: relu_enable = 1 : i32
