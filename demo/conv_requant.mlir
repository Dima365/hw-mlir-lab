module {
  func.func @conv_requant_entry(
      %acc: memref<8x8xi32>,
      %multiplier: memref<8xi32>,
      %shift: memref<8xi32>,
      %out: memref<8x8xi8>)
      attributes {llvm.emit_c_interface} {
    standalone.conv_requant
      ins(%acc, %multiplier, %shift
          : memref<8x8xi32>, memref<8xi32>, memref<8xi32>)
      outs(%out : memref<8x8xi8>)
      {output_zero_point = 17 : i32, relu_enable = 1 : i32}
    return
  }
}
