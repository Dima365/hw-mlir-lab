// RUN: standalone-opt --lower-qadd-relu-to-tiles %s | FileCheck %s --check-prefix=TILES
// RUN: standalone-opt --lower-qadd-relu-to-tiles --one-shot-bufferize="bufferize-function-boundaries" --lower-qadd-relu-to-func-call %s | FileCheck %s --check-prefix=CALLS

module {
  func.func @qadd_tile(
      %lhs: tensor<8x8xui8>,
      %rhs: tensor<8x8xui8>) -> tensor<8x8xui8> {
    %out = tensor.empty() : tensor<8x8xui8>
    %result = standalone.qadd_relu
      ins(%lhs, %rhs : tensor<8x8xui8>, tensor<8x8xui8>)
      outs(%out : tensor<8x8xui8>)
      {
        lhs_multiplier = 17 : i32,
        rhs_multiplier = 13 : i32,
        shift = 4 : i32,
        lhs_zero_point = 68 : i32,
        rhs_zero_point = 65 : i32,
        output_zero_point = 3 : i32,
        relu_enable = 1 : i32
      } -> tensor<8x8xui8>
    return %result : tensor<8x8xui8>
  }
}

// TILES: memref.alloca() : memref<8x8xi8>
// TILES: scf.for
// TILES: tensor.extract
// TILES: memref.store
// TILES: standalone.qadd_relu
// TILES-SAME: ins({{.*}} : memref<8x8xi8>, memref<8x8xi8>)
// TILES-SAME: lhs_zero_point = 68
// TILES-SAME: rhs_zero_point = 65
// TILES: tensor.insert

// CALLS: func.func private @qadd_relu_8x8
// CALLS: call @qadd_relu_8x8
// CALLS-NOT: standalone.qadd_relu
