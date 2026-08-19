// RUN: standalone-opt --transform-interpreter %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %lhs_scale = transform.param.constant 0.25195315503515303 : f64
        -> !transform.any_param
    %rhs_scale = transform.param.constant 0.2558594055008143 : f64
        -> !transform.any_param
    %output_scale = transform.param.constant 1.0 : f64
        -> !transform.any_param
    %lhs_multiplier, %rhs_multiplier, %shift =
        transform.standalone.compute_qadd_fixed_point
            %lhs_scale, %rhs_scale, %output_scale
        : (!transform.any_param, !transform.any_param,
           !transform.any_param) ->
          (!transform.any_param, !transform.any_param,
           !transform.any_param)
    transform.debug.emit_param_as_remark %lhs_multiplier,
        "lhs_multiplier" : !transform.any_param
    transform.debug.emit_param_as_remark %rhs_multiplier,
        "rhs_multiplier" : !transform.any_param
    transform.debug.emit_param_as_remark %shift,
        "shift" : !transform.any_param
    transform.yield
  }
}

// Both scaled coefficients end in exactly .5 at shift 31. RNE keeps the
// coefficient with an even integer part and increments the odd one.
// CHECK: lhs_multiplier 541065280 : i64
// CHECK: rhs_multiplier 549453890 : i64
// CHECK: shift 31 : i64
