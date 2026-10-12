// RUN: standalone-opt --convert-linalg-u8s8-matmul-to-systolic --verify-diagnostics --split-input-file %s

module {
  func.func @unsigned_weights(
      %lhs: memref<8x8xui8>, %rhs: memref<8x8xui8>,
      %acc: memref<8x8xi32>) {
    %zero_point = arith.constant 113 : i32
    // expected-error @+1 {{marked u8s8 matmul was not converted to the systolic op}}
    linalg.generic {
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> ()>,
          affine_map<(m, n, k) -> (m, n)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs, %zero_point
            : memref<8x8xui8>, memref<8x8xui8>, i32)
        outs(%acc : memref<8x8xi32>)
        attrs = {standalone.u8s8_matmul} {
      ^bb0(%lhs_value: ui8, %rhs_value: ui8, %zp: i32, %acc_value: i32):
        %lhs_bits = builtin.unrealized_conversion_cast %lhs_value : ui8 to i8
        %rhs_bits = builtin.unrealized_conversion_cast %rhs_value : ui8 to i8
        %lhs_wide = arith.extui %lhs_bits : i8 to i32
        %rhs_wide = arith.extsi %rhs_bits : i8 to i32
        %lhs_centered = arith.subi %lhs_wide, %zp : i32
        %product = arith.muli %lhs_centered, %rhs_wide : i32
        %sum = arith.addi %acc_value, %product : i32
        linalg.yield %sum : i32
    }
    return
  }
}

// -----

module {
  func.func @missing_zero_point_correction(
      %lhs: memref<8x8xui8>, %rhs: memref<8x8xi8>,
      %acc: memref<8x8xi32>) {
    %zero_point = arith.constant 113 : i32
    // expected-error @+1 {{marked u8s8 matmul was not converted to the systolic op}}
    linalg.generic {
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (k, n)>,
          affine_map<(m, n, k) -> ()>,
          affine_map<(m, n, k) -> (m, n)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%lhs, %rhs, %zero_point
            : memref<8x8xui8>, memref<8x8xi8>, i32)
        outs(%acc : memref<8x8xi32>)
        attrs = {standalone.u8s8_matmul} {
      ^bb0(%lhs_value: ui8, %rhs_value: i8, %zp: i32, %acc_value: i32):
        %lhs_bits = builtin.unrealized_conversion_cast %lhs_value : ui8 to i8
        %lhs_wide = arith.extui %lhs_bits : i8 to i32
        %rhs_wide = arith.extsi %rhs_value : i8 to i32
        %product = arith.muli %lhs_wide, %rhs_wide : i32
        %sum = arith.addi %acc_value, %product : i32
        linalg.yield %sum : i32
    }
    return
  }
}
