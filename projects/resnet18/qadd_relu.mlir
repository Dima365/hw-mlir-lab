module @onnx_qadd_relu_transforms
    attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.consumed})
      -> !transform.any_op {

    %updated_root = transform.foreach_match in %root
        @match_qadd_relu -> @rewrite_qadd_relu,
        @match_qrelu     -> @rewrite_qrelu
        : (!transform.any_op) -> !transform.any_op

    transform.yield %updated_root : !transform.any_op
  }

  // QuantizeLinear(Relu(Add(Dequantize(a), Dequantize(b))))
  transform.named_sequence @match_qadd_relu(
      %candidate: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op,
          !transform.any_op, !transform.any_op,
          !transform.any_op) {

    transform.match.operation_name
        %candidate ["onnx.QuantizeLinear"] : !transform.any_op

    %relu = transform.get_producer_of_operand %candidate[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name
        %relu ["onnx.Relu"] : !transform.any_op

    %add = transform.get_producer_of_operand %relu[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name
        %add ["onnx.Add"] : !transform.any_op

    %lhs_dq = transform.get_producer_of_operand %add[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name
        %lhs_dq ["onnx.DequantizeLinear"] : !transform.any_op

    %rhs_dq = transform.get_producer_of_operand %add[1]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name
        %rhs_dq ["onnx.DequantizeLinear"] : !transform.any_op

    transform.yield %lhs_dq, %rhs_dq, %add, %relu, %candidate
        : !transform.any_op, !transform.any_op,
          !transform.any_op, !transform.any_op,
          !transform.any_op
  }

  transform.named_sequence @match_qrelu(
      %candidate: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op) {

    transform.match.operation_name
        %candidate ["onnx.QuantizeLinear"] : !transform.any_op

    %relu = transform.get_producer_of_operand %candidate[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name
        %relu ["onnx.Relu"] : !transform.any_op

    transform.yield %relu, %candidate
        : !transform.any_op, !transform.any_op
  }

  transform.named_sequence @rewrite_qadd_relu(
      %lhs_dq: !transform.any_op {transform.consumed},
      %rhs_dq: !transform.any_op {transform.consumed},
      %add: !transform.any_op {transform.consumed},
      %relu: !transform.any_op {transform.consumed},
      %quant: !transform.any_op {transform.consumed}) {

    %new_op = transform.standalone.lower_onnx_qadd_relu
        %lhs_dq, %rhs_dq, %add, %relu, %quant
        : (!transform.any_op, !transform.any_op,
           !transform.any_op, !transform.any_op,
           !transform.any_op) -> !transform.any_op

    transform.yield
  }

  transform.named_sequence @rewrite_qrelu(
      %relu: !transform.any_op {transform.consumed},
      %quant: !transform.any_op {transform.consumed}) {

    %new_op = transform.standalone.lower_onnx_qrelu
        %relu, %quant
        : (!transform.any_op, !transform.any_op)
          -> !transform.any_op

    transform.yield
  }
}
