// Simulation top that exposes all accelerator IP blocks through one DUT.
// Each block keeps its own start/done handshake; the socket harness routes a
// command to the matching prefixed port group.
module accelerator_top #(
    parameter int N = 8,
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int PARAM_W = 32,
    parameter int SHIFT_W = 6
) (
    input logic clk,
    input logic rst,

    input  logic                         matmul_start,
    output logic                         matmul_done,
    input  logic [N*N*DATA_W-1:0]        matmul_a_flat,
    input  logic [N*N*DATA_W-1:0]        matmul_b_flat,
    input  logic [N*N*ACC_W-1:0]         matmul_c_in_flat,
    input  logic                         matmul_a_is_unsigned,
    input  logic [DATA_W-1:0]            matmul_a_zero_point,
    output logic [N*N*ACC_W-1:0]         matmul_c_out_flat,

    input  logic                         conv_requant_start,
    output logic                         conv_requant_done,
    input  logic [N*N*ACC_W-1:0]         conv_requant_c_in_flat,
    input  logic [N*ACC_W-1:0]           conv_requant_multiplier_flat,
    input  logic [N*SHIFT_W-1:0]         conv_requant_shift_flat,
    input  logic [DATA_W-1:0]            conv_requant_output_zero_point,
    input  logic                         conv_requant_relu_enable,
    output logic [N*N*DATA_W-1:0]        conv_requant_out_flat,

    input  logic                         qadd_relu_start,
    output logic                         qadd_relu_done,
    input  logic [N*N*DATA_W-1:0]        qadd_relu_lhs_flat,
    input  logic [N*N*DATA_W-1:0]        qadd_relu_rhs_flat,
    input  logic signed [PARAM_W-1:0]    qadd_relu_lhs_multiplier,
    input  logic signed [PARAM_W-1:0]    qadd_relu_rhs_multiplier,
    input  logic [SHIFT_W-1:0]           qadd_relu_shift,
    input  logic [DATA_W-1:0]            qadd_relu_lhs_zero_point,
    input  logic [DATA_W-1:0]            qadd_relu_rhs_zero_point,
    input  logic [DATA_W-1:0]            qadd_relu_output_zero_point,
    input  logic                         qadd_relu_relu_enable,
    output logic [N*N*DATA_W-1:0]        qadd_relu_out_flat
);

    array #(
        .N(N),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) matmul_unit (
        .clk(clk),
        .rst(rst),
        .start(matmul_start),
        .done(matmul_done),
        .a_flat(matmul_a_flat),
        .b_flat(matmul_b_flat),
        .c_in_flat(matmul_c_in_flat),
        .a_is_unsigned(matmul_a_is_unsigned),
        .a_zero_point(matmul_a_zero_point),
        .c_out_flat(matmul_c_out_flat)
    );

    conv_requant #(
        .N(N),
        .ACC_W(ACC_W),
        .OUT_W(DATA_W),
        .SHIFT_W(SHIFT_W)
    ) conv_requant_unit (
        .clk(clk),
        .rst(rst),
        .start(conv_requant_start),
        .done(conv_requant_done),
        .c_in_flat(conv_requant_c_in_flat),
        .multiplier_flat(conv_requant_multiplier_flat),
        .shift_flat(conv_requant_shift_flat),
        .output_zero_point(conv_requant_output_zero_point),
        .relu_enable(conv_requant_relu_enable),
        .out_flat(conv_requant_out_flat)
    );

    qadd_relu #(
        .N(N),
        .DATA_W(DATA_W),
        .PARAM_W(PARAM_W),
        .SHIFT_W(SHIFT_W)
    ) qadd_relu_unit (
        .clk(clk),
        .rst(rst),
        .start(qadd_relu_start),
        .done(qadd_relu_done),
        .lhs_flat(qadd_relu_lhs_flat),
        .rhs_flat(qadd_relu_rhs_flat),
        .lhs_multiplier(qadd_relu_lhs_multiplier),
        .rhs_multiplier(qadd_relu_rhs_multiplier),
        .shift(qadd_relu_shift),
        .lhs_zero_point(qadd_relu_lhs_zero_point),
        .rhs_zero_point(qadd_relu_rhs_zero_point),
        .output_zero_point(qadd_relu_output_zero_point),
        .relu_enable(qadd_relu_relu_enable),
        .out_flat(qadd_relu_out_flat)
    );

endmodule
