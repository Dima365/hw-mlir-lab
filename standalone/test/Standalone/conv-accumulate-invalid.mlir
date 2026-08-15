// RUN: standalone-opt --verify-diagnostics --split-input-file %s

module {
  func.func @bad_group(
      %input: tensor<1x2x4x4xui8>,
      %weight: tensor<2x1x1x1xi8>,
      %bias: tensor<2xi32>) -> tensor<1x2x4x4xi32> {
    // expected-error @+1 {{currently requires group = 1}}
    %acc = standalone.conv_accumulate
      ins(%input, %weight, %bias
          : tensor<1x2x4x4xui8>, tensor<2x1x1x1xi8>, tensor<2xi32>)
      {
        input_zero_point = 0 : i32,
        strides = array<i64: 1, 1>,
        pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        group = 2 : i64
      } -> tensor<1x2x4x4xi32>
    return %acc : tensor<1x2x4x4xi32>
  }
}

// -----

module {
  func.func @bad_channels(
      %input: tensor<1x3x4x4xui8>,
      %weight: tensor<2x2x1x1xi8>,
      %bias: tensor<2xi32>) -> tensor<1x2x4x4xi32> {
    // expected-error @+1 {{requires weight input channels to match input C}}
    %acc = standalone.conv_accumulate
      ins(%input, %weight, %bias
          : tensor<1x3x4x4xui8>, tensor<2x2x1x1xi8>, tensor<2xi32>)
      {
        input_zero_point = 0 : i32,
        strides = array<i64: 1, 1>,
        pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        group = 1 : i64
      } -> tensor<1x2x4x4xi32>
    return %acc : tensor<1x2x4x4xi32>
  }
}

// -----

module {
  func.func @bad_result_shape(
      %input: tensor<1x3x5x7xui8>,
      %weight: tensor<8x3x3x3xi8>,
      %bias: tensor<8xi32>) -> tensor<1x8x5x7xi32> {
    // expected-error @+1 {{result shape does not match convolution geometry}}
    %acc = standalone.conv_accumulate
      ins(%input, %weight, %bias
          : tensor<1x3x5x7xui8>, tensor<8x3x3x3xi8>, tensor<8xi32>)
      {
        input_zero_point = 0 : i32,
        strides = array<i64: 2, 2>,
        pads = array<i64: 1, 1, 1, 1>,
        dilations = array<i64: 1, 1>,
        group = 1 : i64
      } -> tensor<1x8x5x7xi32>
    return %acc : tensor<1x8x5x7xi32>
  }
}

// -----

module {
  func.func @bad_zero_point(
      %input: tensor<1x1x1x1xui8>,
      %weight: tensor<1x1x1x1xi8>,
      %bias: tensor<1xi32>) -> tensor<1x1x1x1xi32> {
    // expected-error @+1 {{requires input_zero_point in [0, 255]}}
    %acc = standalone.conv_accumulate
      ins(%input, %weight, %bias
          : tensor<1x1x1x1xui8>, tensor<1x1x1x1xi8>, tensor<1xi32>)
      {
        input_zero_point = 256 : i32,
        strides = array<i64: 1, 1>,
        pads = array<i64: 0, 0, 0, 0>,
        dilations = array<i64: 1, 1>,
        group = 1 : i64
      } -> tensor<1x1x1x1xi32>
    return %acc : tensor<1x1x1x1xi32>
  }
}
