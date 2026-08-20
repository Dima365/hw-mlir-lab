// 8x8 integer matrix multiply-accumulate tile.
//
// Signed matmul mode:
//   C = C_in + A_i8 * B_i8
//
// Quantized activation mode:
//   C = C_in + (A_u8 - a_zero_point) * B_i8
//
// Inputs and quantization parameters are captured when `start` is accepted and
// may change while the command is busy. Accumulation wraps in ACC_W bits; the
// compiler must prove that a complete operation does not overflow.
module array #(
    parameter int N = 8,
    parameter int DATA_W = 8,
    parameter int ACC_W = 32
) (
    input  logic clk,
    input  logic rst,

    input  logic start,
    output logic done,

    input  logic [N*N*DATA_W-1:0] a_flat,
    input  logic [N*N*DATA_W-1:0] b_flat,
    input  logic [N*N*ACC_W-1:0]  c_in_flat,
    input  logic                  a_is_unsigned,
    input  logic [DATA_W-1:0]     a_zero_point,
    output logic [N*N*ACC_W-1:0]  c_out_flat
);

    localparam int MATRIX_ELEMS = N * N;
    localparam int K_W = (N <= 1) ? 1 : $clog2(N);
    localparam int CENTERED_W = DATA_W + 1;
    localparam int PRODUCT_W = CENTERED_W + DATA_W;

    logic busy;
    logic [K_W-1:0] k;
    logic [N*N*DATA_W-1:0] a_latched;
    logic [N*N*DATA_W-1:0] b_latched;
    logic a_is_unsigned_latched;
    logic [DATA_W-1:0] a_zero_point_latched;

    logic [DATA_W-1:0] a_matrix [0:MATRIX_ELEMS-1];
    logic signed [DATA_W-1:0] b_matrix [0:MATRIX_ELEMS-1];
    logic signed [ACC_W-1:0] c_in_matrix [0:MATRIX_ELEMS-1];
    logic signed [ACC_W-1:0] acc [0:MATRIX_ELEMS-1];

    function automatic logic signed [ACC_W-1:0] multiply_for_acc(
        input logic [DATA_W-1:0] a_value,
        input logic signed [DATA_W-1:0] b_value,
        input logic a_unsigned,
        input logic [DATA_W-1:0] zero_point
    );
        logic signed [CENTERED_W-1:0] a_centered;
        logic signed [PRODUCT_W-1:0] product;
        begin
            if (a_unsigned) begin
                a_centered = $signed({1'b0, a_value}) -
                             $signed({1'b0, zero_point});
            end else begin
                a_centered = $signed({a_value[DATA_W-1], a_value});
            end

            product = a_centered * b_value;
            multiply_for_acc =
                {{(ACC_W-PRODUCT_W){product[PRODUCT_W-1]}}, product};
        end
    endfunction

    generate
        for (genvar idx = 0; idx < MATRIX_ELEMS; idx++) begin : pack_matrix
            assign a_matrix[idx] = a_latched[idx*DATA_W +: DATA_W];
            assign b_matrix[idx] = b_latched[idx*DATA_W +: DATA_W];
            assign c_in_matrix[idx] = c_in_flat[idx*ACC_W +: ACC_W];
            assign c_out_flat[idx*ACC_W +: ACC_W] = acc[idx];
        end
    endgenerate

    initial begin
        if (ACC_W < PRODUCT_W)
            $error("array requires ACC_W >= %0d", PRODUCT_W);
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            done <= 1'b0;
            k <= '0;
            a_latched <= '0;
            b_latched <= '0;
            a_is_unsigned_latched <= 1'b0;
            a_zero_point_latched <= '0;

            for (int idx = 0; idx < MATRIX_ELEMS; idx++) begin
                acc[idx] <= '0;
            end
        end else begin
            done <= 1'b0;

            if (start && !busy) begin
                busy <= 1'b1;
                k <= '0;
                a_latched <= a_flat;
                b_latched <= b_flat;
                a_is_unsigned_latched <= a_is_unsigned;
                a_zero_point_latched <= a_zero_point;

                for (int idx = 0; idx < MATRIX_ELEMS; idx++) begin
                    acc[idx] <= c_in_matrix[idx];
                end
            end else if (busy) begin
                for (int i = 0; i < N; i++) begin
                    for (int j = 0; j < N; j++) begin
                        acc[i*N + j] <= acc[i*N + j] +
                            multiply_for_acc(
                                a_matrix[i*N + int'(k)],
                                b_matrix[int'(k)*N + j],
                                a_is_unsigned_latched,
                                a_zero_point_latched);
                    end
                end

                if (int'(k) == N - 1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end

                k <= k + 1'b1;
            end
        end
    end

endmodule
