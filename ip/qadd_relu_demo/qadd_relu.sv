// Quantized residual addition with optional ReLU.
//
// Both input activation tensors use per-tensor uint8 quantization domains.
// Their centered values are scaled into a common wide fixed-point domain,
// added before rounding, and converted to the output uint8 domain:
//
//   lhs_centered = lhs - lhs_zero_point
//   rhs_centered = rhs - rhs_zero_point
//   wide = lhs_centered * lhs_multiplier
//        + rhs_centered * rhs_multiplier
//   centered = round_to_nearest_even(wide / 2^shift)
//   activated = relu_enable ? max(centered, 0) : centered
//   out = saturate_uint8(activated + output_zero_point)
//
// The multipliers must be non-negative signed PARAM_W-bit values. The shared
// shift is in [0, 63] for the default 64-bit wide datapath. The datapath is
// combinational; `done` mirrors `start` one cycle later.
module qadd_relu #(
    parameter int N = 8,
    parameter int DATA_W = 8,
    parameter int PARAM_W = 32,
    parameter int SHIFT_W = 6
) (
    input  logic clk,
    input  logic rst,
    input  logic start,
    output logic done,

    input  logic [N*N*DATA_W-1:0] lhs_flat,
    input  logic [N*N*DATA_W-1:0] rhs_flat,

    input  logic signed [PARAM_W-1:0] lhs_multiplier,
    input  logic signed [PARAM_W-1:0] rhs_multiplier,
    input  logic        [SHIFT_W-1:0] shift,
    input  logic        [DATA_W-1:0]  lhs_zero_point,
    input  logic        [DATA_W-1:0]  rhs_zero_point,
    input  logic        [DATA_W-1:0]  output_zero_point,
    input  logic relu_enable,

    output logic [N*N*DATA_W-1:0] out_flat
);
    localparam int ELEMS = N * N;
    localparam int WIDE_W = 2 * PARAM_W;
    localparam logic signed [WIDE_W-1:0] UNSIGNED_OUT_MAX =
        $signed((WIDE_W'(1) << DATA_W) - 1'b1);
    localparam logic [DATA_W-1:0] UNSIGNED_MAX_BITS = {DATA_W{1'b1}};

    function automatic logic signed [WIDE_W-1:0] round_shift_rne(
        input logic signed [WIDE_W-1:0] value,
        input logic        [SHIFT_W-1:0] amount
    );
        logic negative;
        logic [WIDE_W-1:0] magnitude;
        logic [WIDE_W-1:0] quotient;
        logic [WIDE_W-1:0] remainder;
        logic [WIDE_W-1:0] half;
        logic [WIDE_W-1:0] mask;
        begin
            if (amount == 0) begin
                round_shift_rne = value;
            end else begin
                negative = value[WIDE_W-1];
                magnitude = negative ? $unsigned(-value) : $unsigned(value);
                quotient = magnitude >> amount;
                half = WIDE_W'(1) << (amount - 1);
                mask = (WIDE_W'(1) << amount) - 1'b1;
                remainder = magnitude & mask;

                if ((remainder > half) ||
                    ((remainder == half) && quotient[0]))
                    quotient = quotient + 1'b1;

                round_shift_rne = negative
                    ? -$signed(quotient)
                    :  $signed(quotient);
            end
        end
    endfunction

    genvar idx;
    generate
      for (idx = 0; idx < ELEMS; idx++) begin : lanes
        logic [DATA_W-1:0] lhs_code;
        logic [DATA_W-1:0] rhs_code;

        logic signed [WIDE_W-1:0] lhs_code_wide;
        logic signed [WIDE_W-1:0] rhs_code_wide;
        logic signed [WIDE_W-1:0] lhs_zero_point_wide;
        logic signed [WIDE_W-1:0] rhs_zero_point_wide;
        logic signed [WIDE_W-1:0] output_zero_point_wide;
        logic signed [WIDE_W-1:0] lhs_centered;
        logic signed [WIDE_W-1:0] rhs_centered;
        logic signed [WIDE_W-1:0] lhs_multiplier_wide;
        logic signed [WIDE_W-1:0] rhs_multiplier_wide;
        logic signed [WIDE_W-1:0] lhs_product;
        logic signed [WIDE_W-1:0] rhs_product;
        logic signed [WIDE_W-1:0] wide;
        logic signed [WIDE_W-1:0] centered;
        logic signed [WIDE_W-1:0] activated;
        logic signed [WIDE_W-1:0] quantized;

        always_comb begin
            lhs_code = lhs_flat[idx*DATA_W +: DATA_W];
            rhs_code = rhs_flat[idx*DATA_W +: DATA_W];

            lhs_code_wide = '0;
            rhs_code_wide = '0;
            lhs_zero_point_wide = '0;
            rhs_zero_point_wide = '0;
            output_zero_point_wide = '0;
            lhs_code_wide[DATA_W-1:0] = lhs_code;
            rhs_code_wide[DATA_W-1:0] = rhs_code;
            lhs_zero_point_wide[DATA_W-1:0] = lhs_zero_point;
            rhs_zero_point_wide[DATA_W-1:0] = rhs_zero_point;
            output_zero_point_wide[DATA_W-1:0] = output_zero_point;

            lhs_centered = lhs_code_wide - lhs_zero_point_wide;
            rhs_centered = rhs_code_wide - rhs_zero_point_wide;
            lhs_multiplier_wide =
                {{(WIDE_W-PARAM_W){lhs_multiplier[PARAM_W-1]}},
                 lhs_multiplier};
            rhs_multiplier_wide =
                {{(WIDE_W-PARAM_W){rhs_multiplier[PARAM_W-1]}},
                 rhs_multiplier};

            lhs_product = lhs_centered * lhs_multiplier_wide;
            rhs_product = rhs_centered * rhs_multiplier_wide;
            wide = lhs_product + rhs_product;
            centered = round_shift_rne(wide, shift);
            activated = (relu_enable && centered < 0) ? '0 : centered;
            quantized = activated + output_zero_point_wide;

            if (quantized < 0)
                out_flat[idx*DATA_W +: DATA_W] = '0;
            else if (quantized > UNSIGNED_OUT_MAX)
                out_flat[idx*DATA_W +: DATA_W] = UNSIGNED_MAX_BITS;
            else
                out_flat[idx*DATA_W +: DATA_W] =
                    quantized[DATA_W-1:0];
        end
      end
    endgenerate

    always_ff @(posedge clk)
      done <= rst ? 1'b0 : start;

endmodule
