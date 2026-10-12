// RUN: standalone-opt --transform-preload-library="transform-library-paths=%S/../../../transforms/conv_im2col.mlir" --transform-interpreter --prepare-u8s8-matmul %s -o %t.logical
// RUN: standalone-opt --fuse-matmul-epilogues %t.logical -o %t.fused
// RUN: standalone-opt --transform-preload-library="transform-library-paths=%S/../../../transforms/u8s8_matmul_tile.mlir" --transform-interpreter %t.fused -o %t.tiled
// RUN: standalone-opt --lower-conv-requant-to-tiles --one-shot-bufferize="bufferize-function-boundaries" --canonicalize --convert-linalg-u8s8-matmul-to-systolic %t.tiled -o %t.hw
// RUN: FileCheck %s --check-prefix=LOGICAL --input-file=%t.logical
// RUN: FileCheck %s --check-prefix=HW --input-file=%t.hw
// RUN: FileCheck %s --check-prefix=TILED --input-file=%t.tiled
// RUN: standalone-opt --lower-systolic-u8s8-to-func-call --lower-conv-requant-to-func-call %t.hw | FileCheck %s --check-prefix=CALLS

module {
  func.func @conv_with_all_tails(
      %input: tensor<1x1x2x3xui8>,
      %weight: tensor<5x1x2x2xi8>,
      %bias: tensor<5xi32>,
      %multiplier: tensor<5xi32>,
      %shift: tensor<5xi32>) -> tensor<1x5x2x3xui8> {
    %acc_input_empty = tensor.empty() : tensor<1x1x2x3xi8>
    %acc_input = linalg.generic {
      indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>, affine_map<(n, c, h, w) -> (n, c, h, w)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%input : tensor<1x1x2x3xui8>) outs(%acc_input_empty : tensor<1x1x2x3xi8>) {
    ^bb0(%a: ui8, %unused: i8):
      %bits = builtin.unrealized_conversion_cast %a : ui8 to i8
      linalg.yield %bits : i8
    } -> tensor<1x1x2x3xi8>
    %acc_zp = arith.constant 113 : i32
    %acc_wzp = arith.constant 0 : i32
    %acc_pad_value = arith.constant 113 : i8
    %acc_padded = tensor.pad %acc_input low[0, 0, 1, 0] high[0, 0, 0, 1] {
    ^bb0(%n: index, %c: index, %h: index, %w: index):
      tensor.yield %acc_pad_value : i8
    } : tensor<1x1x2x3xi8> to tensor<1x1x3x4xi8>
    %acc_empty = tensor.empty() : tensor<1x5x2x3xi32>
    %acc_initial = linalg.broadcast ins(%bias : tensor<5xi32>) outs(%acc_empty : tensor<1x5x2x3xi32>) dimensions = [0, 2, 3]
    %acc = linalg.conv_2d_nchw_fchw_q {
      strides = dense<[1, 1]> : tensor<2xi64>,
      dilations = dense<[1, 1]> : tensor<2xi64>,
      input_cast = #linalg.type_fn<cast_unsigned>,
      weight_cast = #linalg.type_fn<cast_signed>}
      ins(%acc_padded, %weight, %acc_zp, %acc_wzp : tensor<1x1x3x4xi8>, tensor<5x1x2x2xi8>, i32, i32)
      outs(%acc_initial : tensor<1x5x2x3xi32>) -> tensor<1x5x2x3xi32>
    %out = tensor.empty() : tensor<1x5x2x3xui8>
    %result = standalone.conv_requant
      ins(%acc, %multiplier, %shift
          : tensor<1x5x2x3xi32>, tensor<5xi32>, tensor<5xi32>)
      outs(%out : tensor<1x5x2x3xui8>)
      {output_zero_point = 17 : i32, relu_enable = 1 : i32}
      -> tensor<1x5x2x3xui8>
    return %result : tensor<1x5x2x3xui8>
  }
}

// LOGICAL: linalg.broadcast
// LOGICAL: linalg.generic
// LOGICAL: linalg.transpose
// LOGICAL: tensor.collapse_shape
// LOGICAL: tensor.pad
// LOGICAL: tensor.pad
// LOGICAL: tensor.pad
// LOGICAL: attrs = {{ *}}{standalone.u8s8_matmul}
// LOGICAL: arith.extui
// LOGICAL: arith.extsi
// LOGICAL: arith.subi
// LOGICAL: arith.muli
// LOGICAL: arith.addi
// LOGICAL: tensor.extract_slice
// LOGICAL: tensor.expand_shape
// LOGICAL: linalg.transpose
// LOGICAL: standalone.conv_requant
// TILED: scf.for
// TILED: scf.for
// TILED: scf.for
// TILED: tensor.extract_slice {{.*}} : tensor<8x8xi8> to tensor<8x8xi8>
// TILED: attrs = {{ *}}{standalone.u8s8_matmul}
// HW: standalone.systolic_u8s8_matmul
// HW-SAME: lhs_zero_point = 113 : i32
// HW: standalone.conv_requant
// CALLS: call @systolic_u8s8_matmul_8x8
// CALLS: call @conv_requant_8x8
// CALLS-NOT: standalone.
