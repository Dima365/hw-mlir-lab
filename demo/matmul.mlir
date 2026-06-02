module {
  func.func @matmul(
    %arg0: tensor<8x16xi8>,
    %arg1: tensor<16x8xi8>
  ) -> tensor<8x8xi32> {
    %empty = tensor.empty() : tensor<8x8xi32>
    %c0 = arith.constant 0 : i32
    %zero = linalg.fill ins(%c0 : i32)
      outs(%empty : tensor<8x8xi32>) -> tensor<8x8xi32>
    %result = linalg.matmul
      ins(%arg0, %arg1 : tensor<8x16xi8>, tensor<16x8xi8>)
      outs(%zero : tensor<8x8xi32>) -> tensor<8x8xi32>
    return %result : tensor<8x8xi32>
  }
}
