module @onnx_qadd_relu_transforms
    attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %relu_lhs_dq, %relu_rhs_dq, %relu_add, %relu, %relu_quant =
        transform.collect_matching @match_qadd_relu in %root
        : (!transform.any_op) ->
          (!transform.any_op, !transform.any_op, !transform.any_op,
           !transform.any_op, !transform.any_op)

    %direct_lhs_dq, %direct_rhs_dq, %direct_add, %direct_quant =
        transform.collect_matching @match_qadd in %root
        : (!transform.any_op) ->
          (!transform.any_op, !transform.any_op, !transform.any_op,
           !transform.any_op)

    %relu_lhs_scale, %relu_lhs_zero_point,
    %relu_rhs_scale, %relu_rhs_zero_point,
    %relu_output_scale, %relu_output_zero_point =
        transform.standalone.extract_onnx_qadd_qparams %relu_add
        : (!transform.any_op) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param)

    %direct_lhs_scale, %direct_lhs_zero_point,
    %direct_rhs_scale, %direct_rhs_zero_point,
    %direct_output_scale, %direct_output_zero_point =
        transform.standalone.extract_onnx_qadd_qparams %direct_add
        : (!transform.any_op) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param)

    %relu_lhs_multiplier, %relu_rhs_multiplier, %relu_shift =
        transform.standalone.compute_qadd_fixed_point
            %relu_lhs_scale, %relu_rhs_scale, %relu_output_scale
        : (!transform.any_param, !transform.any_param,
           !transform.any_param) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param)

    %direct_lhs_multiplier, %direct_rhs_multiplier, %direct_shift =
        transform.standalone.compute_qadd_fixed_point
            %direct_lhs_scale, %direct_rhs_scale, %direct_output_scale
        : (!transform.any_param, !transform.any_param,
           !transform.any_param) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param)

    transform.debug.emit_remark_at %relu_quant,
        "matched DQ + DQ -> Add -> Relu -> QuantizeLinear"
        : !transform.any_op
    transform.debug.emit_remark_at %direct_quant,
        "matched DQ + DQ -> Add -> QuantizeLinear"
        : !transform.any_op

    %relu_qadds = transform.standalone.lower_onnx_qadd_relu
        %relu_lhs_dq, %relu_rhs_dq, %relu_add, %relu, %relu_quant,
        %relu_lhs_multiplier, %relu_rhs_multiplier, %relu_shift,
        %relu_lhs_zero_point, %relu_rhs_zero_point, %relu_output_zero_point
        : (!transform.any_op, !transform.any_op, !transform.any_op,
           !transform.any_op, !transform.any_op,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param) -> !transform.any_op

    %direct_qadds = transform.standalone.lower_onnx_qadd
        %direct_lhs_dq, %direct_rhs_dq, %direct_add, %direct_quant,
        %direct_lhs_multiplier, %direct_rhs_multiplier, %direct_shift,
        %direct_lhs_zero_point, %direct_rhs_zero_point,
        %direct_output_zero_point
        : (!transform.any_op, !transform.any_op, !transform.any_op,
           !transform.any_op, !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param) -> !transform.any_op

    transform.yield
  }

  transform.named_sequence @match_qadd_relu(
      %candidate: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op, !transform.any_op) {
    transform.match.operation_name %candidate ["onnx.QuantizeLinear"]
        : !transform.any_op

    %relu = transform.get_producer_of_operand %candidate[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %relu ["onnx.Relu"]
        : !transform.any_op

    %add = transform.get_producer_of_operand %relu[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %add ["onnx.Add"]
        : !transform.any_op

    %lhs_dq = transform.get_producer_of_operand %add[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %lhs_dq ["onnx.DequantizeLinear"]
        : !transform.any_op

    %rhs_dq = transform.get_producer_of_operand %add[1]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %rhs_dq ["onnx.DequantizeLinear"]
        : !transform.any_op

    transform.yield %lhs_dq, %rhs_dq, %add, %relu, %candidate
        : !transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op, !transform.any_op
  }

  transform.named_sequence @match_qadd(
      %candidate: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op) {
    transform.match.operation_name %candidate ["onnx.QuantizeLinear"]
        : !transform.any_op

    %add = transform.get_producer_of_operand %candidate[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %add ["onnx.Add"]
        : !transform.any_op

    %lhs_dq = transform.get_producer_of_operand %add[0]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %lhs_dq ["onnx.DequantizeLinear"]
        : !transform.any_op

    %rhs_dq = transform.get_producer_of_operand %add[1]
        : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %rhs_dq ["onnx.DequantizeLinear"]
        : !transform.any_op

    transform.yield %lhs_dq, %rhs_dq, %add, %candidate
        : !transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op
  }
}
