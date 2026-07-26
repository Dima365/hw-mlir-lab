// Per-output-channel convolution requantization.
//
// The input is an N x N tile of final int32 convolution accumulators. Rows are
// spatial positions and columns are output channels. Each column has its own
// fixed-point multiplier and shift, while the uint8 output domain has one
// zero point shared by the complete activation tensor.
//
// For lane (row, channel):
//   scaled = round_to_nearest_even(
//       acc[row, channel] * multiplier[channel] / 2^shift[channel])
//   activated = relu_enable ? max(scaled, 0) : scaled
//   out = saturate_uint8(activated + output_zero_point)
//
// The accumulator must already include bias and any compiler-generated input
// zero-point correction. The datapath is combinational. `done` mirrors `start`
// one cycle later, matching the handshake used by the other demo IP blocks.
module conv_requant #(
    parameter int N = 8,
    parameter int ACC_W = 32,
    parameter int OUT_W = 8,
    parameter int SHIFT_W = 6
) (
    input  logic clk,
    input  logic rst,
    input  logic start,
    output logic done,

    input  logic [N*N*ACC_W-1:0] c_in_flat,
    input  logic [N*ACC_W-1:0] multiplier_flat,
    input  logic [N*SHIFT_W-1:0] shift_flat,
    input  logic [OUT_W-1:0] output_zero_point,
    input  logic relu_enable,

    output logic [N*N*OUT_W-1:0] out_flat
);
    localparam int ELEMS = N * N;
    localparam int PROD_W = 2 * ACC_W;
    localparam logic signed [PROD_W-1:0] UNSIGNED_OUT_MAX =
        $signed((PROD_W'(1) << OUT_W) - 1'b1);
    localparam logic [OUT_W-1:0] UNSIGNED_MAX_BITS = {OUT_W{1'b1}};

    function automatic logic signed [PROD_W-1:0] round_shift_rne(
        input logic signed [PROD_W-1:0] value,
        input logic        [SHIFT_W-1:0] amount
    );
        logic negative;
        logic [PROD_W-1:0] magnitude;
        logic [PROD_W-1:0] quotient;
        logic [PROD_W-1:0] remainder;
        logic [PROD_W-1:0] half;
        logic [PROD_W-1:0] mask;
        begin
            if (amount == 0) begin
                round_shift_rne = value;
            end else begin
                negative = value[PROD_W-1];
                magnitude = negative ? $unsigned(-value) : $unsigned(value);
                quotient = magnitude >> amount;
                half = PROD_W'(1) << (amount - 1);
                mask = (PROD_W'(1) << amount) - 1'b1;
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
        localparam int CHANNEL = idx % N;

        logic signed [ACC_W-1:0] acc;
        logic signed [ACC_W-1:0] multiplier;
        logic        [SHIFT_W-1:0] shift;

        logic signed [PROD_W-1:0] product;
        logic signed [PROD_W-1:0] scaled;
        logic signed [PROD_W-1:0] activated;
        logic signed [PROD_W-1:0] zero_point_wide;
        logic signed [PROD_W-1:0] q;

        always_comb begin
            acc = c_in_flat[idx*ACC_W +: ACC_W];
            multiplier =
                multiplier_flat[CHANNEL*ACC_W +: ACC_W];
            shift = shift_flat[CHANNEL*SHIFT_W +: SHIFT_W];

            product = acc * multiplier;
            scaled = round_shift_rne(product, shift);
            activated = (relu_enable && scaled < 0) ? '0 : scaled;

            zero_point_wide = '0;
            zero_point_wide[OUT_W-1:0] = output_zero_point;
            q = activated + zero_point_wide;

            if (q < 0)
                out_flat[idx*OUT_W +: OUT_W] = '0;
            else if (q > UNSIGNED_OUT_MAX)
                out_flat[idx*OUT_W +: OUT_W] = UNSIGNED_MAX_BITS;
            else
                out_flat[idx*OUT_W +: OUT_W] = q[OUT_W-1:0];
        end
      end
    endgenerate

    always_ff @(posedge clk)
      done <= rst ? 1'b0 : start;

endmodule
