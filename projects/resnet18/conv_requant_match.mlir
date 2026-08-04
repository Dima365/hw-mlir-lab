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

    %relu_activation_scale, %relu_activation_zero_point,
    %relu_weight_scales, %relu_weight_zero_points,
    %relu_output_scale, %relu_output_zero_point,
    %relu_bias_scales, %relu_bias_zero_points =
        transform.standalone.extract_onnx_conv_qparams %conv_with_relu
        : (!transform.any_op) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param)

    %direct_activation_scale, %direct_activation_zero_point,
    %direct_weight_scales, %direct_weight_zero_points,
    %direct_output_scale, %direct_output_zero_point,
    %direct_bias_scales, %direct_bias_zero_points =
        transform.standalone.extract_onnx_conv_qparams %conv_without_relu
        : (!transform.any_op) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param)

    %relu_multipliers, %relu_shifts =
        transform.standalone.compute_conv_requant_fixed_point
            %relu_activation_scale, %relu_weight_scales,
            %relu_weight_zero_points, %relu_output_scale
        : (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param) ->
          (!transform.any_param, !transform.any_param)

    %direct_multipliers, %direct_shifts =
        transform.standalone.compute_conv_requant_fixed_point
            %direct_activation_scale, %direct_weight_scales,
            %direct_weight_zero_points, %direct_output_scale
        : (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param) ->
          (!transform.any_param, !transform.any_param)

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
