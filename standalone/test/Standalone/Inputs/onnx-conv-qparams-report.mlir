module @onnx_conv_qparams_report
    attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %convs = transform.collect_matching @match_conv in %root
        : (!transform.any_op) -> !transform.any_op

    %activation_scale, %activation_zero_point,
    %weight_scales, %weight_zero_points,
    %output_scale, %output_zero_point,
    %bias_scales, %bias_zero_points =
        transform.standalone.extract_onnx_conv_qparams %convs
        : (!transform.any_op) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param)

    %multipliers, %shifts =
        transform.standalone.compute_conv_requant_fixed_point
            %activation_scale, %weight_scales,
            %weight_zero_points, %output_scale
        : (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param) ->
          (!transform.any_param, !transform.any_param)

    transform.debug.emit_param_as_remark %activation_scale,
        "activation_scale" : !transform.any_param
    transform.debug.emit_param_as_remark %activation_zero_point,
        "activation_zero_point" : !transform.any_param
    transform.debug.emit_param_as_remark %weight_scales,
        "weight_scales" : !transform.any_param
    transform.debug.emit_param_as_remark %weight_zero_points,
        "weight_zero_points" : !transform.any_param
    transform.debug.emit_param_as_remark %output_scale,
        "output_scale" : !transform.any_param
    transform.debug.emit_param_as_remark %output_zero_point,
        "output_zero_point" : !transform.any_param
    transform.debug.emit_param_as_remark %bias_scales,
        "bias_scales" : !transform.any_param
    transform.debug.emit_param_as_remark %bias_zero_points,
        "bias_zero_points" : !transform.any_param
    transform.debug.emit_param_as_remark %multipliers,
        "multipliers" : !transform.any_param
    transform.debug.emit_param_as_remark %shifts,
        "shifts" : !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_conv(
      %candidate: !transform.any_op {transform.readonly})
      -> !transform.any_op {
    transform.match.operation_name %candidate ["onnx.Conv"]
        : !transform.any_op
    transform.yield %candidate : !transform.any_op
  }
}
