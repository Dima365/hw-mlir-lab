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
    output logic [N*N*ACC_W-1:0]  c_out_flat
);

    localparam int MATRIX_ELEMS = N * N;
    localparam int K_W = (N <= 1) ? 1 : $clog2(N);

    logic busy;
    logic [K_W-1:0] k;
    logic signed [DATA_W-1:0] a_matrix [0:MATRIX_ELEMS-1];
    logic signed [DATA_W-1:0] b_matrix [0:MATRIX_ELEMS-1];
    logic signed [ACC_W-1:0] a_ext [0:MATRIX_ELEMS-1];
    logic signed [ACC_W-1:0] b_ext [0:MATRIX_ELEMS-1];
    logic signed [ACC_W-1:0] c_in_matrix [0:MATRIX_ELEMS-1];
    logic signed [ACC_W-1:0] acc [0:MATRIX_ELEMS-1];

    generate
        for (genvar idx = 0; idx < MATRIX_ELEMS; idx++) begin : pack_matrix
            assign a_matrix[idx] = a_flat[idx*DATA_W +: DATA_W];
            assign b_matrix[idx] = b_flat[idx*DATA_W +: DATA_W];
            assign c_in_matrix[idx] = c_in_flat[idx*ACC_W +: ACC_W];
            assign c_out_flat[idx*ACC_W +: ACC_W] = acc[idx];

            if (ACC_W > DATA_W) begin : sign_extend
                assign a_ext[idx] = {{(ACC_W-DATA_W){a_matrix[idx][DATA_W-1]}},
                                     a_matrix[idx]};
                assign b_ext[idx] = {{(ACC_W-DATA_W){b_matrix[idx][DATA_W-1]}},
                                     b_matrix[idx]};
            end else begin : truncate
                assign a_ext[idx] = a_matrix[idx][ACC_W-1:0];
                assign b_ext[idx] = b_matrix[idx][ACC_W-1:0];
            end
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            done <= 1'b0;
            k <= '0;

            for (int idx = 0; idx < MATRIX_ELEMS; idx++) begin
                acc[idx] <= '0;
            end
        end else begin
            done <= 1'b0;

            if (start && !busy) begin
                busy <= 1'b1;
                k <= '0;

                for (int idx = 0; idx < MATRIX_ELEMS; idx++) begin
                    acc[idx] <= c_in_matrix[idx];
                end
            end else if (busy) begin
                for (int i = 0; i < N; i++) begin
                    for (int j = 0; j < N; j++) begin
                        acc[i*N + j] <= acc[i*N + j] +
                            a_ext[i*N + int'(k)] * b_ext[int'(k)*N + j];
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
