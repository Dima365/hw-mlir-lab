module {
  func.func @matmul(
    %arg0: tensor<8x16xf32>,
    %arg1: tensor<16x8xf32>
  ) -> tensor<8x8xf32> {
    %empty = tensor.empty() : tensor<8x8xf32>
    %result = linalg.matmul
      ins(%arg0, %arg1 : tensor<8x16xf32>, tensor<16x8xf32>)
      outs(%empty : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %result : tensor<8x8xf32>
  }
}
