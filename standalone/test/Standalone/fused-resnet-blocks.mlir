// RUN: standalone-opt --transform-preload-library="transform-library-paths=%S/../../../transforms/conv_im2col.mlir" --transform-interpreter --prepare-u8s8-matmul --fuse-matmul-epilogues %S/../../../demo/resnet_blocks.mlir -o %t.fused
// RUN: FileCheck %s --check-prefix=SCHEDULE --input-file=%t.fused
// RUN: standalone-opt --transform-preload-library="transform-library-paths=%S/../../../transforms/u8s8_matmul_tile.mlir" --transform-interpreter --lower-conv-requant-to-tiles --lower-qadd-relu-to-tiles --one-shot-bufferize="bufferize-function-boundaries" --canonicalize --cse --convert-linalg-u8s8-matmul-to-systolic %t.fused -o %t.hw
// RUN: FileCheck %s --check-prefix=ALLOC --implicit-check-not='memref<16x8xi32' --implicit-check-not='memref<8x16xi32' --input-file=%t.hw

// Conv1 is materialized, while Conv2 and Add share one position/channel tile.
// SCHEDULE-LABEL: func.func @identity_block
// SCHEDULE: standalone.conv_requant
// SCHEDULE-SAME: relu_enable = 1
// SCHEDULE: %[[MAIN:.*]] = standalone.conv_requant
// SCHEDULE-SAME: relu_enable = 0
// SCHEDULE-SAME: -> tensor<8x8xui8>
// SCHEDULE: standalone.qadd_relu ins(%[[MAIN]],
// SCHEDULE-SAME: tensor<8x8xui8>, tensor<8x8xui8>
// SCHEDULE-SAME: relu_enable = 1

// Each downsample branch retains its own reduction and requantization.
// SCHEDULE-LABEL: func.func @downsample_block
// SCHEDULE: standalone.conv_requant
// SCHEDULE-SAME: relu_enable = 1
// SCHEDULE: %[[MAIN2:.*]] = standalone.conv_requant
// SCHEDULE-SAME: relu_enable = 0
// SCHEDULE: %[[SKIP:.*]] = standalone.conv_requant
// SCHEDULE-SAME: relu_enable = 0
// SCHEDULE: standalone.qadd_relu ins(%[[MAIN2]], %[[SKIP]]
// SCHEDULE-SAME: tensor<8x8xui8>, tensor<8x8xui8>

// No full accumulator buffer. Only Conv1 and the final Add need full matrices;
// the main/skip requantized activations and residual gather stay microtiles.
// ALLOC-LABEL: func.func @identity_block
// ALLOC-COUNT-2: memref.alloc(){{.*}}: memref<16x8xui8>
// ALLOC-NOT: memref.alloc(){{.*}}: memref<16x8xui8>
// ALLOC-LABEL: func.func @downsample_block
// ALLOC-COUNT-2: memref.alloc(){{.*}}: memref<8x16xui8>
// ALLOC-NOT: memref.alloc(){{.*}}: memref<8x16xui8>
