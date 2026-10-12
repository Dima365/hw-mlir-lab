module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %convs = transform.structured.match
      ops{["linalg.conv_2d_nchw_fchw_q"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %packed, %result = transform.structured.convert_conv2d_to_img2col %convs
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
