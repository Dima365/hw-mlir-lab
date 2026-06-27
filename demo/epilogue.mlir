module {
  func.func @requant_entry(%acc: memref<8x8xi32>, %out: memref<8x8xi8>)
      attributes {llvm.emit_c_interface} {
    standalone.requantize ins(%acc : memref<8x8xi32>) outs(%out : memref<8x8xi8>)
      {mult = 12897 : i32, shift = 20 : i32, zero_point = 0 : i32}
    return
  }
}
