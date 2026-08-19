module @onnx_qadd_qparams_report
    attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %adds = transform.collect_matching @match_add in %root
        : (!transform.any_op) -> !transform.any_op

    %lhs_scale, %lhs_zero_point, %rhs_scale, %rhs_zero_point,
    %output_scale, %output_zero_point =
        transform.standalone.extract_onnx_qadd_qparams %adds
        : (!transform.any_op) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param,
           !transform.any_param, !transform.any_param)

    %lhs_multiplier, %rhs_multiplier, %shift =
        transform.standalone.compute_qadd_fixed_point
            %lhs_scale, %rhs_scale, %output_scale
        : (!transform.any_param, !transform.any_param,
           !transform.any_param) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param)

    transform.debug.emit_param_as_remark %lhs_scale,
        "lhs_scale" : !transform.any_param
    transform.debug.emit_param_as_remark %lhs_zero_point,
        "lhs_zero_point" : !transform.any_param
    transform.debug.emit_param_as_remark %rhs_scale,
        "rhs_scale" : !transform.any_param
    transform.debug.emit_param_as_remark %rhs_zero_point,
        "rhs_zero_point" : !transform.any_param
    transform.debug.emit_param_as_remark %output_scale,
        "output_scale" : !transform.any_param
    transform.debug.emit_param_as_remark %output_zero_point,
        "output_zero_point" : !transform.any_param
    transform.debug.emit_param_as_remark %lhs_multiplier,
        "lhs_multiplier" : !transform.any_param
    transform.debug.emit_param_as_remark %rhs_multiplier,
        "rhs_multiplier" : !transform.any_param
    transform.debug.emit_param_as_remark %shift,
        "shift" : !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_add(
      %candidate: !transform.any_op {transform.readonly})
      -> !transform.any_op {
    transform.match.operation_name %candidate ["onnx.Add"]
        : !transform.any_op
    transform.yield %candidate : !transform.any_op
  }
}
