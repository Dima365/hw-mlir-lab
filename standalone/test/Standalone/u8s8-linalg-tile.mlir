// RUN: standalone-opt --transform-preload-library="transform-library-paths=%S/../../../transforms/u8s8_matmul_tile.mlir" --transform-interpreter %s | FileCheck %s --check-prefix=TILED
// RUN: standalone-opt --transform-preload-library="transform-library-paths=%S/../../../transforms/u8s8_matmul_tile.mlir" --transform-interpreter --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-u8s8-matmul-to-systolic %s | FileCheck %s --check-prefix=SYSTOLIC

module {
  func.func @large_u8s8_matmul(
      %lhs: tensor<16x16xui8>,
      %rhs: tensor<16x16xi8>) -> tensor<16x16xi32> {
    %zero_point = arith.constant 113 : i32
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<16x16xi32>
    %initial = linalg.fill ins(%zero : i32)
      outs(%empty : tensor<16x16xi32>) -> tensor<16x16xi32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> ()>,
          affine_map<(m, n, k) -> (m, n)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs, %zero_point
            : tensor<16x16xui8>, tensor<16x16xi8>, i32)
        outs(%initial : tensor<16x16xi32>)
        attrs = {standalone.u8s8_matmul} {
      ^bb0(%lhs_value: ui8, %rhs_value: i8, %zp: i32, %acc_value: i32):
        %lhs_bits = builtin.unrealized_conversion_cast %lhs_value : ui8 to i8
        %lhs_wide = arith.extui %lhs_bits : i8 to i32
        %rhs_wide = arith.extsi %rhs_value : i8 to i32
        %lhs_centered = arith.subi %lhs_wide, %zp : i32
        %product = arith.muli %lhs_centered, %rhs_wide : i32
        %sum = arith.addi %acc_value, %product : i32
        linalg.yield %sum : i32
    } -> tensor<16x16xi32>
    return %result : tensor<16x16xi32>
  }
}

// TILED: scf.for
// TILED: scf.for
// TILED: scf.for
// TILED: tensor.extract_slice {{.*}} : tensor<16x16xui8> to tensor<8x8xui8>
// TILED: tensor.extract_slice {{.*}} : tensor<16x16xi8> to tensor<8x8xi8>
// TILED: linalg.generic
// TILED-SAME: attrs =  {standalone.u8s8_matmul}
// TILED: } -> tensor<8x8xi32>

// SYSTOLIC-NOT: linalg.generic
// SYSTOLIC: standalone.systolic_u8s8_matmul
// SYSTOLIC-SAME: lhs_zero_point = 113 : i32
