// RUN: standalone-opt --verify-diagnostics --split-input-file %s

module {
  func.func @bad_lhs_shape(
      %lhs: memref<8x7xi8>,
      %rhs: memref<8x8xi8>,
      %out: memref<8x8xi8>) {
    // expected-error @+1 {{requires lhs to be memref<8x8xi8>}}
    standalone.qadd_relu
      ins(%lhs, %rhs : memref<8x7xi8>, memref<8x8xi8>)
      outs(%out : memref<8x8xi8>)
      {
        lhs_multiplier = 1 : i32,
        rhs_multiplier = 1 : i32,
        shift = 0 : i32,
        lhs_zero_point = 0 : i32,
        rhs_zero_point = 0 : i32,
        output_zero_point = 0 : i32,
        relu_enable = 0 : i32
      }
    return
  }
}

// -----

module {
  func.func @bad_multiplier(
      %lhs: memref<8x8xi8>,
      %rhs: memref<8x8xi8>,
      %out: memref<8x8xi8>) {
    // expected-error @+1 {{requires non-negative multipliers}}
    standalone.qadd_relu
      ins(%lhs, %rhs : memref<8x8xi8>, memref<8x8xi8>)
      outs(%out : memref<8x8xi8>)
      {
        lhs_multiplier = -1 : i32,
        rhs_multiplier = 1 : i32,
        shift = 0 : i32,
        lhs_zero_point = 0 : i32,
        rhs_zero_point = 0 : i32,
        output_zero_point = 0 : i32,
        relu_enable = 0 : i32
      }
    return
  }
}

// -----

module {
  func.func @bad_shift(
      %lhs: memref<8x8xi8>,
      %rhs: memref<8x8xi8>,
      %out: memref<8x8xi8>) {
    // expected-error @+1 {{requires shift in [0, 63]}}
    standalone.qadd_relu
      ins(%lhs, %rhs : memref<8x8xi8>, memref<8x8xi8>)
      outs(%out : memref<8x8xi8>)
      {
        lhs_multiplier = 1 : i32,
        rhs_multiplier = 1 : i32,
        shift = 64 : i32,
        lhs_zero_point = 0 : i32,
        rhs_zero_point = 0 : i32,
        output_zero_point = 0 : i32,
        relu_enable = 0 : i32
      }
    return
  }
}

// -----

module {
  func.func @bad_zero_point(
      %lhs: memref<8x8xi8>,
      %rhs: memref<8x8xi8>,
      %out: memref<8x8xi8>) {
    // expected-error @+1 {{requires rhs_zero_point in [0, 255]}}
    standalone.qadd_relu
      ins(%lhs, %rhs : memref<8x8xi8>, memref<8x8xi8>)
      outs(%out : memref<8x8xi8>)
      {
        lhs_multiplier = 1 : i32,
        rhs_multiplier = 1 : i32,
        shift = 0 : i32,
        lhs_zero_point = 0 : i32,
        rhs_zero_point = 256 : i32,
        output_zero_point = 0 : i32,
        relu_enable = 0 : i32
      }
    return
  }
}

// -----

module {
  func.func @bad_relu(
      %lhs: memref<8x8xi8>,
      %rhs: memref<8x8xi8>,
      %out: memref<8x8xi8>) {
    // expected-error @+1 {{requires relu_enable to be 0 or 1}}
    standalone.qadd_relu
      ins(%lhs, %rhs : memref<8x8xi8>, memref<8x8xi8>)
      outs(%out : memref<8x8xi8>)
      {
        lhs_multiplier = 1 : i32,
        rhs_multiplier = 1 : i32,
        shift = 0 : i32,
        lhs_zero_point = 0 : i32,
        rhs_zero_point = 0 : i32,
        output_zero_point = 0 : i32,
        relu_enable = 2 : i32
      }
    return
  }
}
