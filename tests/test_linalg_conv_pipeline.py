"""Opt-in numerical checks of the compiled Linalg -> array -> requant path.

Run inside the development container with RUN_LINALG_CONV_TESTS=1.
The Python reference uses direct convolution, not the compiler's im2col.
"""

import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.golden.conv_accumulate import conv_accumulate
from tests.golden.conv_requant import conv_requant_tensor
from tests.golden.qadd_relu import qadd_relu_element


ROOT = Path(__file__).resolve().parents[1]


def flatten(values):
    if not isinstance(values, list):
        return [values]
    return [item for value in values for item in flatten(value)]


def tensor_type(shape, element):
    return "tensor<" + "x".join(map(str, (*shape, element))) + ">"


def shape_of(values):
    shape = []
    while isinstance(values, list):
        shape.append(len(values))
        values = values[0]
    return shape


def make_function(name, inputs, weights, bias, zp, strides, pads, dilations,
                  expected, requant, relu_enable=1, qadd=None,
                  requant_params=None, region_only=False):
    ins_shape, out_shape = shape_of(inputs), shape_of(expected)
    input_type = tensor_type(ins_shape, "i8")
    weight_type = tensor_type(shape_of(weights), "i8")
    acc_type = tensor_type(out_shape, "i32")
    bias_type = tensor_type([len(bias)], "i32")
    padded_shape = [*ins_shape[:2], ins_shape[2] + pads[0] + pads[2],
                    ins_shape[3] + pads[1] + pads[3]]
    padded_type = tensor_type(padded_shape, "i8")
    out_type = tensor_type(out_shape, "ui8") if requant else acc_type
    result = "%acc"
    epilogue = ""
    if requant:
        channels = out_shape[1]
        multipliers = [channel + 1 for channel in range(channels)]
        shifts = [9 + channel % 3 for channel in range(channels)]
        output_zp = 37
        if requant_params is not None:
            multipliers, shifts, output_zp = requant_params
        epilogue = f"""
    %mult = arith.constant dense<{json.dumps(multipliers)}> : {bias_type}
    %shift = arith.constant dense<{json.dumps(shifts)}> : {bias_type}
    %out = tensor.empty() : {out_type}
    %q = standalone.conv_requant
      ins(%acc, %mult, %shift : {acc_type}, {bias_type}, {bias_type})
      outs(%out : {out_type})
      {{output_zero_point = {output_zp} : i32, relu_enable = {relu_enable} : i32}} -> {out_type}
"""
        result = "%q"
        if qadd is not None:
            residual, swapped, add_relu = qadd
            residual_type = tensor_type(out_shape, "i8")
            lhs, rhs = ("%residual", "%q") if swapped else ("%q", "%residual")
            lhs_zp, rhs_zp = (113, output_zp) if swapped else (output_zp, 113)
            epilogue += f"""
    %residual_bits = arith.constant dense<{json.dumps(residual)}> : {residual_type}
    %residual_out = tensor.empty() : {out_type}
    %residual = linalg.generic {{
      indexing_maps = [affine_map<(n,c,h,w)->(n,c,h,w)>, affine_map<(n,c,h,w)->(n,c,h,w)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}}
      ins(%residual_bits : {residual_type}) outs(%residual_out : {out_type}) {{
    ^bb0(%bits: i8, %unused: ui8):
      %unsigned = builtin.unrealized_conversion_cast %bits : i8 to ui8
      linalg.yield %unsigned : ui8
    }} -> {out_type}
    %sum_out = tensor.empty() : {out_type}
    %sum = standalone.qadd_relu ins({lhs}, {rhs} : {out_type}, {out_type})
      outs(%sum_out : {out_type})
      {{lhs_multiplier = 3 : i32, rhs_multiplier = 5 : i32, shift = 3 : i32,
        lhs_zero_point = {lhs_zp} : i32, rhs_zero_point = {rhs_zp} : i32,
        output_zero_point = 91 : i32, relu_enable = {add_relu} : i32}} -> {out_type}
"""
            result = "%sum"
    body = f"""
    %weights = arith.constant dense<{json.dumps(weights)}> : {weight_type}
    %bias = arith.constant dense<{json.dumps(bias)}> : {bias_type}
    %zp = arith.constant {zp} : i32
    %wzp = arith.constant 0 : i32
    %pad_value = arith.constant {zp} : i8
    %padded = tensor.pad %input
      low[0, 0, {pads[0]}, {pads[1]}] high[0, 0, {pads[2]}, {pads[3]}] {{
    ^bb0(%n: index, %c: index, %h: index, %w: index):
      tensor.yield %pad_value : i8
    }} : {input_type} to {padded_type}
    %empty = tensor.empty() : {acc_type}
    %initial = linalg.broadcast ins(%bias : {bias_type})
      outs(%empty : {acc_type}) dimensions = [0, 2, 3]
    %acc = linalg.conv_2d_nchw_fchw_q {{
      strides = dense<{json.dumps(strides)}> : tensor<2xi64>,
      dilations = dense<{json.dumps(dilations)}> : tensor<2xi64>,
      input_cast = #linalg.type_fn<cast_unsigned>,
      weight_cast = #linalg.type_fn<cast_signed>}}
      ins(%padded, %weights, %zp, %wzp : {padded_type}, {weight_type}, i32, i32)
      outs(%initial : {acc_type}) -> {acc_type}
    {epilogue}
    {"scf.yield" if region_only else "return"} {result} : {out_type}
"""
    if region_only:
        return f"scf.execute_region -> {out_type} {{\n{body}\n}}"
    return f"func.func @{name}(%input: {input_type}) -> {out_type} {{\n{body}\n}}"


def make_driver(cases):
    declarations = []
    checks = []
    for name, inputs, expected, requant in cases:
        ctype = "uint8_t" if requant else "int32_t"
        ins_shape, out_shape = shape_of(inputs), shape_of(expected)
        def descriptor(data, shape):
            strides = [math.prod(shape[index + 1:]) for index in range(4)]
            sizes = ", ".join(map(str, shape))
            steps = ", ".join(map(str, strides))
            return "{" + f"{data}, {data}, 0, {{{sizes}}}, {{{steps}}}" + "}"
        declarations.append(
            f"extern void _mlir_ciface_{name}_entry(Ref *, Ref *);")
        expected_flat = flatten(expected)
        checks.append(f"""
  {{
    uint8_t input[] = {{{", ".join(map(str, flatten(inputs)))}}};
    {ctype} output[{len(expected_flat)}] = {{0}};
    {ctype} expected[] = {{{", ".join(map(str, expected_flat))}}};
    Ref in = {descriptor("input", ins_shape)};
    Ref out = {descriptor("output", out_shape)};
    _mlir_ciface_{name}_entry(&in, &out);
    for (int i = 0; i < {len(expected_flat)}; ++i) {{
      if (output[i] != expected[i]) {{
        fprintf(stderr, "{name}[%d]: got %d expected %d\\n",
                i, (int)output[i], (int)expected[i]);
        return 1;
      }}
    }}
  }}
""")
    return """
#include <stdint.h>
#include <stdio.h>
typedef struct {
  void *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
} Ref;
""" + "\n".join(declarations) + "\nint main(void) {\n" + "".join(checks) + """
  puts("PASS Linalg Conv accumulators and per-channel requantization");
  return 0;
}
"""


@unittest.skipUnless(os.environ.get("RUN_LINALG_CONV_TESTS") == "1",
                     "requires MLIR tools and cocotb/Verilator")
class LinalgConvPipelineTest(unittest.TestCase):
    def check_compiled_module(self, functions, cases):
        with tempfile.TemporaryDirectory(prefix="linalg-conv-") as directory:
            tmp = Path(directory)
            mlir, driver = tmp / "conv.mlir", tmp / "main.c"
            mlir.write_text("module {\n" + "".join(functions) + "}\n")
            driver.write_text(make_driver(cases))

            def run(*args, env=None):
                result = subprocess.run(args, cwd=ROOT, text=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=300,
                                        env=env)
                self.assertEqual(result.returncode, 0, result.stdout)

            inlined = tmp / "inlined.mlir"
            opt = os.environ.get("OPT", str(ROOT / "standalone/build/bin/standalone-opt"))
            run(opt, "--canonicalize", str(mlir), "-o", str(inlined))
            run("bash", "pipelines/resnet_blocks_pipeline.sh", str(inlined),
                str(tmp))
            fused = (tmp / "01_fused_epilogues.mlir").read_text()
            requants = [line for line in fused.splitlines()
                        if "standalone.conv_requant" in line]
            self.assertTrue(requants)
            for line in requants:
                self.assertIn("tensor<8x8xi32>", line)
            for line in fused.splitlines():
                if "standalone.qadd_relu" in line:
                    self.assertIn("tensor<8x8xui8>", line)
            run("bash", "pipelines/compile_pipeline.sh",
                str(tmp / "resnet_blocks.ll"), str(driver),
                str(tmp / "app"), str(tmp))
            run("bash", "pipelines/cocotb_pipeline.sh",
                str(tmp / "app"), "accelerator")

    def test_compiled_convolutions_match_python(self):
        functions, cases = [], []
        # Batch > 1, asymmetric padding, stride/dilation, M/N/K tails,
        # several K tiles, boundary zero points, and all byte bit patterns.
        configs = [
            (2, 3, 4, 5, 5, 2, 3, 113, [1, 2], [1, 2, 0, 1], [1, 1]),
            (1, 2, 5, 6, 9, 3, 2, 128, [2, 1], [2, 0, 1, 2], [2, 2]),
            (1, 1, 2, 3, 3, 1, 1, 255, [1, 1], [0, 0, 0, 0], [1, 1]),
            (1, 2, 2, 4, 8, 1, 1, 0, [1, 1], [0, 0, 0, 0], [1, 1]),
        ]
        for index, (batch, cin, h, w, cout, kh, kw, zp, stride, pad, dil) in enumerate(configs):
            inputs = [[[[((n * 53 + c * 71 + y * 37 + x * 97) % 256)
                         for x in range(w)] for y in range(h)]
                       for c in range(cin)] for n in range(batch)]
            weights = [[[[((f * 31 + c * 11 + y * 13 + x * 7) % 256) - 128
                          for x in range(kw)] for y in range(kh)]
                        for c in range(cin)] for f in range(cout)]
            bias = [f * 251 - 613 for f in range(cout)]
            acc = conv_accumulate(inputs, weights, bias, zp, strides=stride,
                                  pads=pad, dilations=dil)
            quantized = conv_requant_tensor(
                acc, list(range(1, cout + 1)),
                [9 + c % 3 for c in range(cout)], 37, index % 2)
            for requant, expected in [(False, acc), (True, quantized)]:
                name = f"case_{index}_{int(requant)}"
                functions.append(make_function(
                    name, inputs, weights, bias, zp, stride, pad, dil,
                    expected, requant, relu_enable=index % 2))
                cases.append((name, inputs, expected, requant))
            # Main may be either operand; the parameter sets must not swap
            # accidentally. Batch*H*W is not necessarily a multiple of eight.
            n_count, channels, out_h, out_w = shape_of(quantized)
            residual = [[[[((n * 43 + c * 67 + y * 31 + x * 89) % 256)
                            for x in range(out_w)] for y in range(out_h)]
                          for c in range(channels)] for n in range(n_count)]
            swapped, add_relu = bool(index % 2), (index // 2) % 2
            def add_element(main, skip):
                lhs, rhs = (skip, main) if swapped else (main, skip)
                lhs_zp, rhs_zp = (113, 37) if swapped else (37, 113)
                return qadd_relu_element(lhs, rhs, 3, 5, 3, lhs_zp, rhs_zp,
                                         91, add_relu)
            expected = [[[[add_element(quantized[n][c][y][x], residual[n][c][y][x])
                           for x in range(out_w)] for y in range(out_h)]
                         for c in range(channels)] for n in range(n_count)]
            name = f"case_{index}_add"
            functions.append(make_function(
                name, inputs, weights, bias, zp, stride, pad, dil, expected,
                True, relu_enable=index % 2, qadd=(residual, swapped, add_relu)))
            cases.append((name, inputs, expected, True))
        self.check_compiled_module(functions, cases)

    def test_downsample_with_distinct_branch_parameters(self):
        inputs = [[[[((n * 53 + c * 71 + y * 37 + x * 97) % 256)
                     for x in range(5)] for y in range(3)]
                   for c in range(3)] for n in range(2)]
        branches, results = [], []
        for branch, kernel, zp, out_zp, relu in [
                ("main", 3, 71, 37, 0), ("skip", 1, 113, 113, 1)]:
            weights = [[[[((f * 2 + c + y + x) % 7) - 3
                          for x in range(kernel)] for y in range(kernel)]
                        for c in range(3)] for f in range(5)]
            bias = [c * 13 - 71 for c in range(5)]
            pads = [kernel // 2] * 4
            acc = conv_accumulate(inputs, weights, bias, zp,
                                  strides=[2, 2], pads=pads)
            multipliers = [c + kernel for c in range(5)]
            shifts = [kernel + 2 + c % 3 for c in range(5)]
            result = conv_requant_tensor(acc, multipliers, shifts, out_zp, relu)
            results.append(result)
            branches.append(make_function(
                branch, inputs, weights, bias, zp, [2, 2], pads, [1, 1],
                result, True, relu_enable=relu,
                requant_params=(multipliers, shifts, out_zp), region_only=True))
        main, skip = results
        expected = [[[[qadd_relu_element(skip[n][c][y][x], main[n][c][y][x],
                                        3, 5, 3, 113, 37, 91, 0)
                       for x in range(3)] for y in range(2)]
                     for c in range(5)] for n in range(2)]
        functions = ["""
  func.func @dual_branch(%input: tensor<2x3x3x5xi8>) -> tensor<2x5x2x3xui8> {
""" + f"%main = {branches[0]}\n%skip = {branches[1]}\n" + """
    %out = tensor.empty() : tensor<2x5x2x3xui8>
    %sum = standalone.qadd_relu
      ins(%skip, %main : tensor<2x5x2x3xui8>, tensor<2x5x2x3xui8>)
      outs(%out : tensor<2x5x2x3xui8>)
      {lhs_multiplier = 3 : i32, rhs_multiplier = 5 : i32, shift = 3 : i32,
       lhs_zero_point = 113 : i32, rhs_zero_point = 37 : i32,
       output_zero_point = 91 : i32, relu_enable = 0 : i32}
      -> tensor<2x5x2x3xui8>
    return %sum : tensor<2x5x2x3xui8>
  }
"""]
        self.check_compiled_module(functions,
                                   [("dual_branch", inputs, expected, True)])


if __name__ == "__main__":
    unittest.main()
