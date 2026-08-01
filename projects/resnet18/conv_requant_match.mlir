module @onnx_conv_requant_match
    attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %conv_with_relu, %relu, %quant_after_relu =
        transform.collect_matching @match_conv_relu_quantize in %root
        : (!transform.any_op) ->
          (!transform.any_op, !transform.any_op, !transform.any_op)

    %conv_without_relu, %quant_after_conv =
        transform.collect_matching @match_conv_quantize in %root
        : (!transform.any_op) ->
          (!transform.any_op, !transform.any_op)

    transform.debug.emit_remark_at %quant_after_relu,
        "matched Conv -> Relu -> QuantizeLinear" : !transform.any_op
    transform.debug.emit_remark_at %quant_after_conv,
        "matched Conv -> QuantizeLinear" : !transform.any_op

    transform.yield
  }

  transform.named_sequence @match_conv_relu_quantize(
      %candidate: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.any_op) {
    transform.match.operation_name %candidate ["onnx.QuantizeLinear"]
        : !transform.any_op

    %relu = transform.get_producer_of_operand %candidate[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %relu ["onnx.Relu"]
        : !transform.any_op

    %conv = transform.get_producer_of_operand %relu[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %conv ["onnx.Conv"]
        : !transform.any_op

    transform.yield %conv, %relu, %candidate
        : !transform.any_op, !transform.any_op, !transform.any_op
  }

  transform.named_sequence @match_conv_quantize(
      %candidate: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op) {
    transform.match.operation_name %candidate ["onnx.QuantizeLinear"]
        : !transform.any_op

    %conv = transform.get_producer_of_operand %candidate[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %conv ["onnx.Conv"]
        : !transform.any_op

    transform.yield %conv, %candidate
        : !transform.any_op, !transform.any_op
  }
}
