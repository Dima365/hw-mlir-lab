module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op

    %padded, %pad, %copy_back = transform.structured.pad %matmuls
      pad_to_multiple_of [8, 8, 8] {
        padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32],
        padding_dimensions = [0, 1, 2],
        nofold_flags = [1, 1, 0]
      } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.yield
  }
}
