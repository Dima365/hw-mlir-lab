// RUN: standalone-opt --verify-diagnostics --split-input-file --lower-conv-requant-to-tiles --lower-qadd-relu-to-tiles %s

func.func @nchw_requant(%acc: tensor<1x8x2x8xi32>,
    %mult: tensor<8xi32>, %shift: tensor<8xi32>) -> tensor<1x8x2x8xui8> {
  %out = tensor.empty() : tensor<1x8x2x8xui8>
  // expected-error @+1 {{expected a fused 8x8 matrix tile; run fuse-matmul-epilogues before hardware buffer lowering}}
  %q = standalone.conv_requant ins(%acc, %mult, %shift
      : tensor<1x8x2x8xi32>, tensor<8xi32>, tensor<8xi32>)
    outs(%out : tensor<1x8x2x8xui8>)
    {output_zero_point = 7 : i32, relu_enable = 0 : i32} -> tensor<1x8x2x8xui8>
  return %q : tensor<1x8x2x8xui8>
}

// -----

func.func @untiled_requant(%acc: tensor<16x8xi32>,
    %mult: tensor<8xi32>, %shift: tensor<8xi32>) -> tensor<16x8xui8> {
  %out = tensor.empty() : tensor<16x8xui8>
  // expected-error @+1 {{expected a fused 8x8 matrix tile; run fuse-matmul-epilogues before hardware buffer lowering}}
  %q = standalone.conv_requant ins(%acc, %mult, %shift
      : tensor<16x8xi32>, tensor<8xi32>, tensor<8xi32>)
    outs(%out : tensor<16x8xui8>)
    {output_zero_point = 7 : i32, relu_enable = 0 : i32} -> tensor<16x8xui8>
  return %q : tensor<16x8xui8>
}

// -----

func.func @nchw_add(%lhs: tensor<1x8x2x8xui8>,
    %rhs: tensor<1x8x2x8xui8>) -> tensor<1x8x2x8xui8> {
  %out = tensor.empty() : tensor<1x8x2x8xui8>
  // expected-error @+1 {{expected a fused 8x8 matrix tile; run fuse-matmul-epilogues before hardware buffer lowering}}
  %sum = standalone.qadd_relu ins(%lhs, %rhs : tensor<1x8x2x8xui8>, tensor<1x8x2x8xui8>)
    outs(%out : tensor<1x8x2x8xui8>)
    {lhs_multiplier = 3 : i32, rhs_multiplier = 5 : i32, shift = 4 : i32,
     lhs_zero_point = 7 : i32, rhs_zero_point = 113 : i32,
     output_zero_point = 29 : i32, relu_enable = 1 : i32} -> tensor<1x8x2x8xui8>
  return %sum : tensor<1x8x2x8xui8>
}

// -----

func.func @untiled_add(%lhs: tensor<16x8xui8>,
    %rhs: tensor<16x8xui8>) -> tensor<16x8xui8> {
  %out = tensor.empty() : tensor<16x8xui8>
  // expected-error @+1 {{expected a fused 8x8 matrix tile; run fuse-matmul-epilogues before hardware buffer lowering}}
  %sum = standalone.qadd_relu ins(%lhs, %rhs : tensor<16x8xui8>, tensor<16x8xui8>)
    outs(%out : tensor<16x8xui8>)
    {lhs_multiplier = 3 : i32, rhs_multiplier = 5 : i32, shift = 4 : i32,
     lhs_zero_point = 7 : i32, rhs_zero_point = 113 : i32,
     output_zero_point = 29 : i32, relu_enable = 1 : i32} -> tensor<16x8xui8>
  return %sum : tensor<16x8xui8>
}
