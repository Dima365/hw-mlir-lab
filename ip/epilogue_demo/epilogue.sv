// Epilogue IP (v1): per-tensor requantize + ReLU.
//
// For each of N*N elements:
//   prod = c_in * mult + round_add        (round half up; PROD_W intermediate)
//   q    = (prod >>> shift) + zero_point
//   out  = clamp(q, 0, OUT_MAX)           (lower bound 0 == ReLU)
//
// Combinational datapath; `done` mirrors `start` one cycle later to match the
// start/done handshake used by the other IP blocks.
module epilogue #(
    parameter int N = 8,
    parameter int ACC_W = 32,
    parameter int OUT_W = 8
) (
    input  logic clk,
    input  logic rst,
    input  logic start,
    output logic done,

    input  logic [N*N*ACC_W-1:0]    c_in_flat,
    input  logic signed [ACC_W-1:0] mult,
    input  logic [4:0]              shift,
    input  logic signed [ACC_W-1:0] zero_point,
    output logic [N*N*OUT_W-1:0]    out_flat
);
    localparam int ELEMS = N * N;
    localparam int PROD_W = 2 * ACC_W;
    localparam int OUT_MAX = (1 << (OUT_W - 1)) - 1;  // 127 for OUT_W=8

    logic signed [PROD_W-1:0] round_add;
    assign round_add = (shift == 0) ? '0 : (PROD_W'(1) <<< (shift - 1));

    genvar idx;
    generate
      for (idx = 0; idx < ELEMS; idx++) begin : lanes
        logic signed [ACC_W-1:0]  c_in;
        logic signed [PROD_W-1:0] prod;
        logic signed [ACC_W-1:0]  q;

        assign c_in = c_in_flat[idx*ACC_W +: ACC_W];
        assign prod = c_in * mult + round_add;             // PROD_W intermediate
        assign q    = ACC_W'(prod >>> shift) + zero_point; // arithmetic shift

        assign out_flat[idx*OUT_W +: OUT_W] =
            (q < 0)       ? '0 :
            (q > OUT_MAX) ? OUT_W'(OUT_MAX) : q[OUT_W-1:0];
      end
    endgenerate

    always_ff @(posedge clk)
      done <= rst ? 1'b0 : start;

endmodule
