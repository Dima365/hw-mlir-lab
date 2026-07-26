// RUN: standalone-opt --verify-diagnostics --split-input-file %s

module {
  func.func @bad_acc_shape(
      %acc: memref<8x7xi32>,
      %multiplier: memref<8xi32>,
      %shift: memref<8xi32>,
      %out: memref<8x8xi8>) {
    // expected-error @+1 {{requires acc to be memref<8x8xi32>}}
    standalone.conv_requant
      ins(%acc, %multiplier, %shift
          : memref<8x7xi32>, memref<8xi32>, memref<8xi32>)
      outs(%out : memref<8x8xi8>)
      {output_zero_point = 0 : i32, relu_enable = 0 : i32}
    return
  }
}

// -----

module {
  func.func @bad_output_zero_point(
      %acc: memref<8x8xi32>,
      %multiplier: memref<8xi32>,
      %shift: memref<8xi32>,
      %out: memref<8x8xi8>) {
    // expected-error @+1 {{requires output_zero_point in [0, 255]}}
    standalone.conv_requant
      ins(%acc, %multiplier, %shift
          : memref<8x8xi32>, memref<8xi32>, memref<8xi32>)
      outs(%out : memref<8x8xi8>)
      {output_zero_point = 256 : i32, relu_enable = 0 : i32}
    return
  }
}

// -----

module {
  func.func @bad_relu_enable(
      %acc: memref<8x8xi32>,
      %multiplier: memref<8xi32>,
      %shift: memref<8xi32>,
      %out: memref<8x8xi8>) {
    // expected-error @+1 {{requires relu_enable to be 0 or 1}}
    standalone.conv_requant
      ins(%acc, %multiplier, %shift
          : memref<8x8xi32>, memref<8xi32>, memref<8xi32>)
      outs(%out : memref<8x8xi8>)
      {output_zero_point = 0 : i32, relu_enable = 2 : i32}
    return
  }
}
