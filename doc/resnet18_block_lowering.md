# ResNet18 Block Lowering

`projects/resnet18/lower_resnet18_blocks.sh` lowers the quantized Conv and
residual portions of the ONNX QDQ ResNet18 graph to the three accelerator IP
contracts.

## Pipeline

1. `conv_requant_match.mlir` replaces supported ONNX Conv QDQ chains with
   `linalg.conv_2d_nchw_fchw_q -> standalone.conv_requant`. Standard
   `tensor.pad` fills spatial padding with the activation zero point, and
   `linalg.broadcast` initializes the i32 accumulator with the channel bias.
   The convolution explicitly selects unsigned input and signed weight casts.
2. `qadd_relu.mlir` replaces residual Add QDQ chains with tensor-form
   `standalone.qadd_relu`.
3. `transforms/conv_im2col.mlir` invokes the fork's standard
   `transform.structured.convert_conv2d_to_img2col` operation.
   `--prepare-u8s8-matmul` validates the resulting contraction, transposes and
   reshapes its operands to positions-by-channels order, and invokes the
   standard Linalg padding utility. The resulting large matrix multiplication
   is represented by a marked tensor-form `linalg.generic`.
4. `--fuse-matmul-epilogues` normalizes supported epilogues to padded `[M, C]`
   matrices and uses SCF `TilingInterface` fusion to share their 8x8 output
   tile loops with matmul. `transforms/u8s8_matmul_tile.mlir` then uses
   Transform Dialect to split each
   logical matmul into `8 positions x 8 output channels x 8 reduction values`
   microkernels.
5. `--lower-conv-requant-to-tiles` and `--lower-qadd-relu-to-tiles` convert
   already tiled rank-2 8x8 epilogues to hardware buffers. Full NCHW tensors and
   untiled matrices are rejected; there is no independent epilogue tiling path.
6. One-shot bufferization turns the tiled tensor computation into memrefs and
   strided subviews.
7. `--convert-linalg-u8s8-matmul-to-systolic` validates and replaces the exact
   marked Linalg microkernels with `standalone.systolic_u8s8_matmul`.
8. The three runtime passes replace hardware tile operations with calls to the
   C bridge in `interface/interface.c`.

Run from the repository root:

```bash
make lower-resnet18-blocks
```

Intermediate files are written to `build/resnet18-block-lowering/`.

To build and simulate the self-contained identity and downsample block demo:

```bash
make demo-resnet-blocks
```

This command lowers `demo/resnet_blocks.mlir`, compiles the generated LLVM IR
with the C runtime bridge, and runs the application against the combined
`accelerator` RTL top through cocotb/Verilator. The application checks both
block outputs against fixed results produced from the Python golden model.

## Conv Tile Layout

For NCHW input and OIHW weights, the fork's im2col produces an input tensor
`[batch, K, positions]`, weights `[channels, K]`, and output
`[batch, channels, positions]`. Linalg transposes and Tensor reshapes normalize
these to the logical matrix multiplication:

```text
A[M_pad, K_pad] x B[K_pad, N_pad] + C[M_pad, N_pad]

M = batch * output_height * output_width
K = input_channels * kernel_height * kernel_width
N = output_channels
```

`M_pad`, `K_pad`, and `N_pad` are rounded up to multiples of eight. Transform
Dialect then tiles this operation. One array invocation computes:

```text
A[8 positions, 8 K] x B[8 K, 8 output channels]
  + C[8 positions, 8 output channels]
```

The reduction index is:

```text
k = (input_channel * kernel_height + kernel_row) * kernel_width
    + kernel_column
```

The logical accumulator matrix is initialized from `bias[output_channel]`.
The marked `linalg.generic` explicitly performs unsigned extension of
activations, signed extension of weights, activation zero-point subtraction,
multiplication and accumulation. The reduction loops produced by Transform
Dialect carry the accumulator tile between consecutive K tiles.

The separate Linalg-to-accelerator pass validates this arithmetic and replaces
it with `standalone.systolic_u8s8_matmul`, whose RTL contract performs the same
zero-point correction. Spatial padding and K-tail activation lanes contain the
activation zero point, so they contribute exactly zero. Weight and
output-channel tails contain zero weights. Output-channel tails use zero bias;
invalid spatial lanes are discarded at the output boundary.

After the full reduction, `standalone.conv_requant` applies per-output-channel
multiplier/shift values and writes uint8 codes. A fused QADD consumes that
uint8 tile before the final result is restored to NCHW order. Both rounding
and saturation boundaries remain; this is schedule fusion, not arithmetic
fusion that skips Conv quantization.

The implementation still materializes complete padded im2col and weight
matrices. For fused chains, bias is broadcast directly into a tile and the
full i32 accumulator is eliminated. Padded rows may contain broadcast bias;
they are discarded at the output boundary and cannot affect valid elements.
Intermediate main/skip uint8 activations before a fused QADD are also tile-local.
Packing copies at the synchronous runtime boundary remain.

## Shared Output Tiles

The epilogue tensor forms support rank-4 NCHW and rank-2 `[M, C]` shapes,
`DestinationStyleOpInterface`, and fixed-size `TilingInterface` tiling.
Channel parameters are sliced only along C. The normalized matrices are padded
to multiples of eight before tiling, so hardware tiles remain statically 8x8.
For residual gathers, M is decoded as `(batch, height, width)` even when a tile
crosses a batch boundary; padded residual lanes use that operand's zero point.

The shared schedule is:

```text
for each 8-position x 8-channel output tile:
    acc = channel bias
    for each K tile:
        acc = ARRAY(A_tile, B_tile, acc)
    main = CONV_REQUANT(acc, channel multipliers, channel shifts)
    if Add is present:
        skip = residual tile or independently computed downsample tile
        result = QADD_RELU(main, skip)
    else:
        result = main
    store result tile
```

ReLU flags on both IPs are copied from the actual graph; neither Add nor ReLU
is mandatory. Both operand orders of QADD are supported without exchanging
their quantization parameters. Downsample and main retain separate K loops,
bias values and requantization. Tile destinations are allocated outside the
shared loops and reused only after synchronous calls complete.

Fusion recognizes the exact layout bridge produced by
`--prepare-u8s8-matmul` and requires channel-broadcast bias and single-use
accumulators. A requantized input to QADD must also have a single use.
Unsupported producer/layout/bias chains, shared intermediate inputs to QADD,
and QADD without a fusible matmul/requant producer are compilation errors.
An identity residual is gathered from its existing tensor; a downsample
residual has its own fused matmul/requant producer.

These single-use checks do not restrict boundary activations that feed the
main and residual branches of the next block. The checks apply to intermediate
values fused into an epilogue, not to every activation in the graph.

Conv1 and Conv2 are not fused with each other: the full uint8 output of Conv1
is still retained for the next spatial convolution. No RTL or socket protocol
changes are required, and no direct streaming connection between IPs is implied.

Both pipeline scripts always run fusion. There is no switch to the former
independent NCHW schedule. After fusion, every tensor epilogue must be a
rank-2 8x8 tile. NCHW tensor forms remain valid frontend IR, but must pass
through fusion before hardware buffer lowering.

Inspect `01_fused_epilogues.mlir` in the demo output directory or
`03_fused_epilogues.mlir` in the full-graph output directory. Bufferized IR
should contain only 8x8 i32 accumulator buffers for successfully fused chains.

## Residual Blocks

An identity block naturally becomes:

```text
Conv1 tiles -> CONV_REQUANT(ReLU)
Conv2 tiles -> CONV_REQUANT
identity input ---------------------> QADD_RELU
```

A downsample block adds a third Conv path:

```text
main Conv1 -> CONV_REQUANT(ReLU) -> main Conv2 -> CONV_REQUANT
input -> 1x1/stride-2 Conv -> CONV_REQUANT --------------------> QADD_RELU
```

No monolithic ResNet-block operation is required. SSA dependencies retain the
branch structure and order the synchronous runtime calls.

The combined simulation top instantiates the systolic array, `CONV_REQUANT`,
and `QADD_RELU`. The socket protocol opcode selects the target IP for every
runtime request.

## Current Boundary

The Docker toolchain pins Dima365/llvm-project commit
`bc11b5680d41ea81de5822568166a470f1ea26bf`, including quantized NCHW im2col and
independent input/weight cast attributes. Rebuild the image and standalone
tool after updating from the previous toolchain. There is no
`standalone.conv_accumulate` operation or project-specific im2col algorithm
in the compiler. Historical generated IR such as
`projects/resnet18/resnet18_integer_conv.mlir` must be regenerated with the
current frontend before use; the full-graph script writes fresh IR under
`build/resnet18-block-lowering/`.

The hardware path requires static shapes, group 1, uint8 activations with a
constant scalar zero point, signed int8 weights with zero zero points, and
i32 accumulation. Per-channel weight scales remain in CONV_REQUANT's
multiplier/shift arrays. Nonzero weight zero points and signed activation
extension are rejected, not silently mapped to array instructions.

The script lowers the Conv and residual subgraphs but intentionally leaves
other ONNX operations, such as input quantization, max-pooling, average-pooling
and the classifier, in ONNX dialect. A complete executable ResNet18 still needs
lowering for those operations and a top-level memory/runtime integration.

## Numerical Regression

Inside the rebuilt development container:

```bash
RUN_LINALG_CONV_TESTS=1 python3 -m unittest tests.test_linalg_conv_pipeline -v
```

This compiles and simulates the fused lowering against a
direct Python convolution, per-channel requantization, and quantized Add.
Cases cover batch 2,
nonzero bias, asymmetric padding, stride and dilation, values above 127,
negative weights, zero points 0/113/128/255, M/N/K tails, optional Add, both
QADD operand orders, distinct input scales/zero points, and enabled/disabled
ReLU on both IPs. Lit tests check rejection of unsupported shared consumers,
missing producers/bias, and untiled epilogues, as well as parameter slicing,
direct tile connections, and the absence of full accumulator allocations.
The test is skipped
in ordinary host-only Python discovery unless explicitly enabled.
