// RUN: standalone-opt --verify-diagnostics --split-input-file %s

module {
  func.func @bad_lhs(
      %lhs: memref<8x8xsi8>,
      %rhs: memref<8x8xi8>,
      %acc: memref<8x8xi32>) {
    // expected-error @+1 {{requires lhs to be memref<8x8xi8> or memref<8x8xui8>}}
    standalone.systolic_u8s8_matmul
      ins(%lhs, %rhs : memref<8x8xsi8>, memref<8x8xi8>)
      outs(%acc : memref<8x8xi32>)
      {lhs_zero_point = 0 : i32}
    return
  }
}

// -----

module {
  func.func @bad_rhs(
      %lhs: memref<8x8xui8>,
      %rhs: memref<8x8xui8>,
      %acc: memref<8x8xi32>) {
    // expected-error @+1 {{requires rhs to be memref<8x8xi8>}}
    standalone.systolic_u8s8_matmul
      ins(%lhs, %rhs : memref<8x8xui8>, memref<8x8xui8>)
      outs(%acc : memref<8x8xi32>)
      {lhs_zero_point = 0 : i32}
    return
  }
}

// -----

module {
  func.func @bad_zero_point(
      %lhs: memref<8x8xui8>,
      %rhs: memref<8x8xi8>,
      %acc: memref<8x8xi32>) {
    // expected-error @+1 {{requires lhs_zero_point in [0, 255]}}
    standalone.systolic_u8s8_matmul
      ins(%lhs, %rhs : memref<8x8xui8>, memref<8x8xi8>)
      outs(%acc : memref<8x8xi32>)
      {lhs_zero_point = 256 : i32}
    return
  }
}
