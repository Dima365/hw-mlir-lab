// Configurable quantized epilogue.
//
// For each of N*N elements, the fixed pipeline is:
//   main     = (c_in + optional_bias) * mult
//   residual = (residual_in - residual_zero_point) * residual_mult
//   combined = main + (add_enable ? residual : 0)
//   centered = round_to_nearest_even(combined / 2^shift)
//   activated = relu_enable ? max(centered, 0) : centered
//   q = activated + zero_point
//   out = saturate(q, signed/unsigned OUT_W range)
//
// Both products use the same binary point (`shift`), so residual addition is
// performed in a common wide domain and rounded only once. Bias values are in
// the accumulator domain. The main accumulator is already zero-centered;
// residual_in is an unsigned quantized activation and is centered explicitly.
//
// Scheduling constraint: `mult` and `shift` are scalar parameters, so one
// invocation must process elements from a single output channel. Processing
// multiple channels in one invocation requires per-channel multipliers and
// shifts (unless those channels happen to use identical quantization scales).
//
// Fusion semantics: this epilogue consumes the int32 convolution accumulator
// and performs bias, optional residual add, optional ReLU, and output
// quantization as one fused operation. It therefore rounds to uint8/int8 only
// after the fused Conv epilogue. This can retain more precision than an ONNX
// QDQ graph that requantizes the Conv result before Add, but it is not bit-exact
// with that graph and must be validated against the intended model accuracy.
//
// The datapath is combinational. `done` mirrors `start` one cycle later to
// match the handshake used by the other demo IP blocks.
module epilogue #(
    parameter int N = 8,
    parameter int ACC_W = 32,
    parameter int OUT_W = 8
) (
    input  logic clk,
    input  logic rst,
    input  logic start,
    output logic done,

    input  logic [N*N*ACC_W-1:0] c_in_flat,
    input  logic [N*N*ACC_W-1:0] bias_in_flat,
    input  logic [N*N*OUT_W-1:0] residual_in_flat,

    input  logic signed [ACC_W-1:0] mult,
    input  logic signed [ACC_W-1:0] residual_mult,
    input  logic        [5:0]       shift,
    input  logic signed [ACC_W-1:0] residual_zero_point,
    input  logic signed [ACC_W-1:0] zero_point,

    input  logic bias_enable,
    input  logic add_enable,
    input  logic relu_enable,
    input  logic output_signed,

    output logic [N*N*OUT_W-1:0] out_flat
);
    localparam int ELEMS = N * N;
    // One extra bit for acc+bias and one for adding the residual product.
    localparam int WIDE_W = 2 * ACC_W + 2;
    localparam logic signed [WIDE_W-1:0] SIGNED_OUT_MIN =
        -$signed(WIDE_W'(1) << (OUT_W - 1));
    localparam logic signed [WIDE_W-1:0] SIGNED_OUT_MAX =
         $signed((WIDE_W'(1) << (OUT_W - 1)) - 1'b1);
    localparam logic signed [WIDE_W-1:0] UNSIGNED_OUT_MAX =
         $signed((WIDE_W'(1) << OUT_W) - 1'b1);

    localparam logic [OUT_W-1:0] SIGNED_MIN_BITS =
        {1'b1, {(OUT_W-1){1'b0}}};
    localparam logic [OUT_W-1:0] SIGNED_MAX_BITS =
        {1'b0, {(OUT_W-1){1'b1}}};
    localparam logic [OUT_W-1:0] UNSIGNED_MAX_BITS = {OUT_W{1'b1}};

    function automatic logic signed [WIDE_W-1:0] round_shift_rne(
        input logic signed [WIDE_W-1:0] value,
        input logic        [5:0]        amount
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
        logic signed [ACC_W-1:0] c_in;
        logic signed [ACC_W-1:0] bias_in;
        logic        [OUT_W-1:0] residual_in;

        logic signed [WIDE_W-1:0] main_value;
        logic signed [WIDE_W-1:0] residual_value;
        logic signed [WIDE_W-1:0] mult_wide;
        logic signed [WIDE_W-1:0] residual_mult_wide;
        logic signed [WIDE_W-1:0] main_product;
        logic signed [WIDE_W-1:0] residual_product;
        logic signed [WIDE_W-1:0] combined;
        logic signed [WIDE_W-1:0] centered;
        logic signed [WIDE_W-1:0] activated;
        logic signed [WIDE_W-1:0] q;

        always_comb begin
            c_in = c_in_flat[idx*ACC_W +: ACC_W];
            bias_in = bias_in_flat[idx*ACC_W +: ACC_W];
            residual_in = residual_in_flat[idx*OUT_W +: OUT_W];

            main_value = {{(WIDE_W-ACC_W){c_in[ACC_W-1]}}, c_in};
            if (bias_enable)
                main_value = main_value +
                    {{(WIDE_W-ACC_W){bias_in[ACC_W-1]}}, bias_in};

            residual_value = WIDE_W'($unsigned(residual_in));
            residual_value = residual_value -
                {{(WIDE_W-ACC_W){residual_zero_point[ACC_W-1]}},
                  residual_zero_point};

            mult_wide = {{(WIDE_W-ACC_W){mult[ACC_W-1]}}, mult};
            residual_mult_wide =
                {{(WIDE_W-ACC_W){residual_mult[ACC_W-1]}}, residual_mult};

            main_product = main_value * mult_wide;
            residual_product = residual_value * residual_mult_wide;
            combined = main_product;
            if (add_enable)
                combined = combined + residual_product;

            centered = round_shift_rne(combined, shift);
            activated = (relu_enable && centered < 0) ? '0 : centered;
            q = activated +
                {{(WIDE_W-ACC_W){zero_point[ACC_W-1]}}, zero_point};

            if (output_signed) begin
                if (q < SIGNED_OUT_MIN)
                    out_flat[idx*OUT_W +: OUT_W] = SIGNED_MIN_BITS;
                else if (q > SIGNED_OUT_MAX)
                    out_flat[idx*OUT_W +: OUT_W] = SIGNED_MAX_BITS;
                else
                    out_flat[idx*OUT_W +: OUT_W] = q[OUT_W-1:0];
            end else begin
                if (q < 0)
                    out_flat[idx*OUT_W +: OUT_W] = '0;
                else if (q > UNSIGNED_OUT_MAX)
                    out_flat[idx*OUT_W +: OUT_W] = UNSIGNED_MAX_BITS;
                else
                    out_flat[idx*OUT_W +: OUT_W] = q[OUT_W-1:0];
            end
        end
      end
    endgenerate

    always_ff @(posedge clk)
      done <= rst ? 1'b0 : start;

endmodule
