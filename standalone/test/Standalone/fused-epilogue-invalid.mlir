// RUN: standalone-opt --verify-diagnostics --split-input-file --transform-preload-library="transform-library-paths=%S/../../../transforms/conv_im2col.mlir" --transform-interpreter --prepare-u8s8-matmul --fuse-matmul-epilogues %s

func.func @shared_acc(%input: tensor<1x8x2x8xi8>, %weight: tensor<8x8x1x1xi8>,
                      %bias: tensor<8xi32>, %mult: tensor<8xi32>, %shift: tensor<8xi32>)
    -> (tensor<1x8x2x8xi32>, tensor<1x8x2x8xui8>) {
  %zp = arith.constant 113 : i32
  %zero = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<1x8x2x8xi32>
  %initial = linalg.broadcast ins(%bias : tensor<8xi32>)
    outs(%empty : tensor<1x8x2x8xi32>) dimensions = [0, 2, 3]
  %acc = linalg.conv_2d_nchw_fchw_q {
    strides = dense<[1, 1]> : tensor<2xi64>, dilations = dense<[1, 1]> : tensor<2xi64>,
    input_cast = #linalg.type_fn<cast_unsigned>, weight_cast = #linalg.type_fn<cast_signed>}
    ins(%input, %weight, %zp, %zero : tensor<1x8x2x8xi8>, tensor<8x8x1x1xi8>, i32, i32)
    outs(%initial : tensor<1x8x2x8xi32>) -> tensor<1x8x2x8xi32>
  %out = tensor.empty() : tensor<1x8x2x8xui8>
  // expected-error @+1 {{fusion requires a single-use prepared matmul accumulator}}
  %q = standalone.conv_requant ins(%acc, %mult, %shift
      : tensor<1x8x2x8xi32>, tensor<8xi32>, tensor<8xi32>)
    outs(%out : tensor<1x8x2x8xui8>)
    {output_zero_point = 7 : i32, relu_enable = 0 : i32} -> tensor<1x8x2x8xui8>
  return %acc, %q : tensor<1x8x2x8xi32>, tensor<1x8x2x8xui8>
}

// -----

func.func @shared_activation(%input: tensor<1x8x2x8xi8>,
    %weight: tensor<8x8x1x1xi8>, %bias: tensor<8xi32>,
    %mult: tensor<8xi32>, %shift: tensor<8xi32>, %skip: tensor<1x8x2x8xui8>)
    -> (tensor<1x8x2x8xui8>, tensor<1x8x2x8xui8>) {
  %zp = arith.constant 113 : i32
  %zero = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<1x8x2x8xi32>
  %initial = linalg.broadcast ins(%bias : tensor<8xi32>)
    outs(%empty : tensor<1x8x2x8xi32>) dimensions = [0, 2, 3]
  %acc = linalg.conv_2d_nchw_fchw_q {
    strides = dense<[1, 1]> : tensor<2xi64>, dilations = dense<[1, 1]> : tensor<2xi64>,
    input_cast = #linalg.type_fn<cast_unsigned>, weight_cast = #linalg.type_fn<cast_signed>}
    ins(%input, %weight, %zp, %zero : tensor<1x8x2x8xi8>, tensor<8x8x1x1xi8>, i32, i32)
    outs(%initial : tensor<1x8x2x8xi32>) -> tensor<1x8x2x8xi32>
  %out = tensor.empty() : tensor<1x8x2x8xui8>
  // expected-error @+1 {{requantized input to QADD must have a single use}}
  %q = standalone.conv_requant ins(%acc, %mult, %shift
      : tensor<1x8x2x8xi32>, tensor<8xi32>, tensor<8xi32>)
    outs(%out : tensor<1x8x2x8xui8>)
    {output_zero_point = 7 : i32, relu_enable = 1 : i32} -> tensor<1x8x2x8xui8>
  %sum = standalone.qadd_relu ins(%q, %skip : tensor<1x8x2x8xui8>, tensor<1x8x2x8xui8>)
    outs(%out : tensor<1x8x2x8xui8>)
    {lhs_multiplier = 3 : i32, rhs_multiplier = 5 : i32, shift = 4 : i32,
     lhs_zero_point = 7 : i32, rhs_zero_point = 113 : i32,
     output_zero_point = 29 : i32, relu_enable = 0 : i32} -> tensor<1x8x2x8xui8>
  return %q, %sum : tensor<1x8x2x8xui8>, tensor<1x8x2x8xui8>
}

// -----

func.func @missing_producer(%acc: tensor<1x8x2x8xi32>,
    %mult: tensor<8xi32>, %shift: tensor<8xi32>) -> tensor<1x8x2x8xui8> {
  %out = tensor.empty() : tensor<1x8x2x8xui8>
  // expected-error @+1 {{fusion requires a single-use prepared matmul accumulator}}
  %q = standalone.conv_requant ins(%acc, %mult, %shift
      : tensor<1x8x2x8xi32>, tensor<8xi32>, tensor<8xi32>)
    outs(%out : tensor<1x8x2x8xui8>)
    {output_zero_point = 7 : i32, relu_enable = 0 : i32} -> tensor<1x8x2x8xui8>
  return %q : tensor<1x8x2x8xui8>
}

// -----

func.func @missing_bias(%input: tensor<1x8x2x8xi8>,
    %weight: tensor<8x8x1x1xi8>, %initial: tensor<1x8x2x8xi32>,
    %mult: tensor<8xi32>, %shift: tensor<8xi32>) -> tensor<1x8x2x8xui8> {
  %zp = arith.constant 113 : i32
  %zero = arith.constant 0 : i32
  %acc = linalg.conv_2d_nchw_fchw_q {
    input_cast = #linalg.type_fn<cast_unsigned>, weight_cast = #linalg.type_fn<cast_signed>}
    ins(%input, %weight, %zp, %zero : tensor<1x8x2x8xi8>, tensor<8x8x1x1xi8>, i32, i32)
    outs(%initial : tensor<1x8x2x8xi32>) -> tensor<1x8x2x8xi32>
  %out = tensor.empty() : tensor<1x8x2x8xui8>
  // expected-error @+1 {{fusion requires channel-broadcast bias initialization}}
  %q = standalone.conv_requant ins(%acc, %mult, %shift
      : tensor<1x8x2x8xi32>, tensor<8xi32>, tensor<8xi32>)
    outs(%out : tensor<1x8x2x8xui8>)
    {output_zero_point = 7 : i32, relu_enable = 0 : i32} -> tensor<1x8x2x8xui8>
  return %q : tensor<1x8x2x8xui8>
}

// -----

func.func @standalone_add(%lhs: tensor<1x8x2x8xui8>,
    %rhs: tensor<1x8x2x8xui8>) -> tensor<1x8x2x8xui8> {
  %out = tensor.empty() : tensor<1x8x2x8xui8>
  // expected-error @+1 {{QADD fusion requires at least one single-use matmul -> conv_requant producer}}
  %sum = standalone.qadd_relu ins(%lhs, %rhs : tensor<1x8x2x8xui8>, tensor<1x8x2x8xui8>)
    outs(%out : tensor<1x8x2x8xui8>)
    {lhs_multiplier = 3 : i32, rhs_multiplier = 5 : i32, shift = 4 : i32,
     lhs_zero_point = 7 : i32, rhs_zero_point = 113 : i32,
     output_zero_point = 29 : i32, relu_enable = 1 : i32} -> tensor<1x8x2x8xui8>
  return %sum : tensor<1x8x2x8xui8>
}

// -----

func.func @untiled_matrix(%acc: tensor<16x8xi32>,
    %mult: tensor<8xi32>, %shift: tensor<8xi32>) -> tensor<16x8xui8> {
  %out = tensor.empty() : tensor<16x8xui8>
  // expected-error @+1 {{fusion must produce an 8x8 matrix epilogue tile}}
  %q = standalone.conv_requant ins(%acc, %mult, %shift
      : tensor<16x8xi32>, tensor<8xi32>, tensor<8xi32>)
    outs(%out : tensor<16x8xui8>)
    {output_zero_point = 7 : i32, relu_enable = 0 : i32} -> tensor<16x8xui8>
  return %q : tensor<16x8xui8>
}
