// RUN: not standalone-opt --transform-preload-library="transform-library-paths=%S/../../../transforms/conv_im2col.mlir" --transform-interpreter --prepare-u8s8-matmul %s 2>&1 | FileCheck %s
module {
  func.func @unsupported(%input: tensor<1x1x3x3xi8>,
                         %weight: tensor<1x1x1x1xi8>,
                         %init: tensor<1x1x3x3xi32>) -> tensor<1x1x3x3xi32> {
    %izp = arith.constant 113 : i32
    %wzp = arith.constant -2 : i32
    %r = linalg.conv_2d_nchw_fchw_q {
      input_cast = #linalg.type_fn<cast_unsigned>,
      weight_cast = #linalg.type_fn<cast_signed>}
      ins(%input, %weight, %izp, %wzp : tensor<1x1x3x3xi8>, tensor<1x1x1x1xi8>, i32, i32)
      outs(%init : tensor<1x1x3x3xi32>) -> tensor<1x1x3x3xi32>
    return %r : tensor<1x1x3x3xi32>
  }
}
// CHECK: error: systolic array requires weight zero point = 0
