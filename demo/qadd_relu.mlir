module {
  func.func @qadd_relu_entry(
      %lhs: memref<8x8xi8>,
      %rhs: memref<8x8xi8>,
      %out: memref<8x8xi8>)
      attributes {llvm.emit_c_interface} {
    standalone.qadd_relu
      ins(%lhs, %rhs : memref<8x8xi8>, memref<8x8xi8>)
      outs(%out : memref<8x8xi8>)
      {
        lhs_multiplier = 1715842648 : i32,
        rhs_multiplier = 1328001668 : i32,
        shift = 30 : i32,
        lhs_zero_point = 68 : i32,
        rhs_zero_point = 65 : i32,
        output_zero_point = 0 : i32,
        relu_enable = 1 : i32
      }
    return
  }
}
