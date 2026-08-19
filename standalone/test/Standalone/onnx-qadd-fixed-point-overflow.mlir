// RUN: not standalone-opt --transform-interpreter %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %lhs_scale = transform.param.constant 3.0e9 : f64
        -> !transform.any_param
    %rhs_scale = transform.param.constant 1.0 : f64
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
    transform.yield
  }
}

// CHECK: QAdd scale set at index 0: scale ratios do not fit signed i32 multipliers
