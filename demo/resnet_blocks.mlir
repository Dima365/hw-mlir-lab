module {
  func.func @identity_block(%input: tensor<1x8x4x4xui8>)
      -> tensor<1x8x4x4xui8> {
    %weight1 = arith.constant dense<1> : tensor<8x8x3x3xi8>
    %weight2 = arith.constant dense<-1> : tensor<8x8x3x3xi8>
    %bias = arith.constant dense<0> : tensor<8xi32>
    %multiplier = arith.constant dense<1> : tensor<8xi32>
    %shift4 = arith.constant dense<4> : tensor<8xi32>

    %acc1_input_empty = tensor.empty() : tensor<1x8x4x4xi8>
    %acc1_input = linalg.generic {
      indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>, affine_map<(n, c, h, w) -> (n, c, h, w)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%input : tensor<1x8x4x4xui8>) outs(%acc1_input_empty : tensor<1x8x4x4xi8>) {
    ^bb0(%a: ui8, %unused: i8):
      %bits = builtin.unrealized_conversion_cast %a : ui8 to i8
      linalg.yield %bits : i8
    } -> tensor<1x8x4x4xi8>
    %acc1_zp = arith.constant 100 : i32
    %acc1_wzp = arith.constant 0 : i32
    %acc1_pad_value = arith.constant 100 : i8
    %acc1_padded = tensor.pad %acc1_input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%n: index, %c: index, %h: index, %w: index):
      tensor.yield %acc1_pad_value : i8
    } : tensor<1x8x4x4xi8> to tensor<1x8x6x6xi8>
    %acc1_empty = tensor.empty() : tensor<1x8x4x4xi32>
    %acc1_initial = linalg.broadcast ins(%bias : tensor<8xi32>) outs(%acc1_empty : tensor<1x8x4x4xi32>) dimensions = [0, 2, 3]
    %acc1 = linalg.conv_2d_nchw_fchw_q {
      strides = dense<[1, 1]> : tensor<2xi64>,
      dilations = dense<[1, 1]> : tensor<2xi64>,
      input_cast = #linalg.type_fn<cast_unsigned>,
      weight_cast = #linalg.type_fn<cast_signed>}
      ins(%acc1_padded, %weight1, %acc1_zp, %acc1_wzp : tensor<1x8x6x6xi8>, tensor<8x8x3x3xi8>, i32, i32)
      outs(%acc1_initial : tensor<1x8x4x4xi32>) -> tensor<1x8x4x4xi32>
    %empty1 = tensor.empty() : tensor<1x8x4x4xui8>
    %conv1 = standalone.conv_requant
      ins(%acc1, %multiplier, %shift4
          : tensor<1x8x4x4xi32>, tensor<8xi32>, tensor<8xi32>)
      outs(%empty1 : tensor<1x8x4x4xui8>)
      {output_zero_point = 100 : i32, relu_enable = 1 : i32}
      -> tensor<1x8x4x4xui8>

    %acc2_input_empty = tensor.empty() : tensor<1x8x4x4xi8>
    %acc2_input = linalg.generic {
      indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>, affine_map<(n, c, h, w) -> (n, c, h, w)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%conv1 : tensor<1x8x4x4xui8>) outs(%acc2_input_empty : tensor<1x8x4x4xi8>) {
    ^bb0(%a: ui8, %unused: i8):
      %bits = builtin.unrealized_conversion_cast %a : ui8 to i8
      linalg.yield %bits : i8
    } -> tensor<1x8x4x4xi8>
    %acc2_zp = arith.constant 100 : i32
    %acc2_wzp = arith.constant 0 : i32
    %acc2_pad_value = arith.constant 100 : i8
    %acc2_padded = tensor.pad %acc2_input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%n: index, %c: index, %h: index, %w: index):
      tensor.yield %acc2_pad_value : i8
    } : tensor<1x8x4x4xi8> to tensor<1x8x6x6xi8>
    %acc2_empty = tensor.empty() : tensor<1x8x4x4xi32>
    %acc2_initial = linalg.broadcast ins(%bias : tensor<8xi32>) outs(%acc2_empty : tensor<1x8x4x4xi32>) dimensions = [0, 2, 3]
    %acc2 = linalg.conv_2d_nchw_fchw_q {
      strides = dense<[1, 1]> : tensor<2xi64>,
      dilations = dense<[1, 1]> : tensor<2xi64>,
      input_cast = #linalg.type_fn<cast_unsigned>,
      weight_cast = #linalg.type_fn<cast_signed>}
      ins(%acc2_padded, %weight2, %acc2_zp, %acc2_wzp : tensor<1x8x6x6xi8>, tensor<8x8x3x3xi8>, i32, i32)
      outs(%acc2_initial : tensor<1x8x4x4xi32>) -> tensor<1x8x4x4xi32>
    %empty2 = tensor.empty() : tensor<1x8x4x4xui8>
    %conv2 = standalone.conv_requant
      ins(%acc2, %multiplier, %shift4
          : tensor<1x8x4x4xi32>, tensor<8xi32>, tensor<8xi32>)
      outs(%empty2 : tensor<1x8x4x4xui8>)
      {output_zero_point = 100 : i32, relu_enable = 0 : i32}
      -> tensor<1x8x4x4xui8>

    %empty3 = tensor.empty() : tensor<1x8x4x4xui8>
    %result = standalone.qadd_relu
      ins(%conv2, %input
          : tensor<1x8x4x4xui8>, tensor<1x8x4x4xui8>)
      outs(%empty3 : tensor<1x8x4x4xui8>)
      {
        lhs_multiplier = 1 : i32,
        rhs_multiplier = 1 : i32,
        shift = 0 : i32,
        lhs_zero_point = 100 : i32,
        rhs_zero_point = 100 : i32,
        output_zero_point = 100 : i32,
        relu_enable = 1 : i32
      } -> tensor<1x8x4x4xui8>
    return %result : tensor<1x8x4x4xui8>
  }

  func.func @downsample_block(%input: tensor<1x8x4x4xui8>)
      -> tensor<1x16x2x2xui8> {
    %main_weight1 = arith.constant dense<1> : tensor<16x8x3x3xi8>
    %main_weight2 = arith.constant dense<-1> : tensor<16x16x3x3xi8>
    %skip_weight = arith.constant dense<1> : tensor<16x8x1x1xi8>
    %bias = arith.constant dense<0> : tensor<16xi32>
    %multiplier = arith.constant dense<1> : tensor<16xi32>
    %shift4 = arith.constant dense<4> : tensor<16xi32>

    %main_acc1_input_empty = tensor.empty() : tensor<1x8x4x4xi8>
    %main_acc1_input = linalg.generic {
      indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>, affine_map<(n, c, h, w) -> (n, c, h, w)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%input : tensor<1x8x4x4xui8>) outs(%main_acc1_input_empty : tensor<1x8x4x4xi8>) {
    ^bb0(%a: ui8, %unused: i8):
      %bits = builtin.unrealized_conversion_cast %a : ui8 to i8
      linalg.yield %bits : i8
    } -> tensor<1x8x4x4xi8>
    %main_acc1_zp = arith.constant 100 : i32
    %main_acc1_wzp = arith.constant 0 : i32
    %main_acc1_pad_value = arith.constant 100 : i8
    %main_acc1_padded = tensor.pad %main_acc1_input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%n: index, %c: index, %h: index, %w: index):
      tensor.yield %main_acc1_pad_value : i8
    } : tensor<1x8x4x4xi8> to tensor<1x8x6x6xi8>
    %main_acc1_empty = tensor.empty() : tensor<1x16x2x2xi32>
    %main_acc1_initial = linalg.broadcast ins(%bias : tensor<16xi32>) outs(%main_acc1_empty : tensor<1x16x2x2xi32>) dimensions = [0, 2, 3]
    %main_acc1 = linalg.conv_2d_nchw_fchw_q {
      strides = dense<[2, 2]> : tensor<2xi64>,
      dilations = dense<[1, 1]> : tensor<2xi64>,
      input_cast = #linalg.type_fn<cast_unsigned>,
      weight_cast = #linalg.type_fn<cast_signed>}
      ins(%main_acc1_padded, %main_weight1, %main_acc1_zp, %main_acc1_wzp : tensor<1x8x6x6xi8>, tensor<16x8x3x3xi8>, i32, i32)
      outs(%main_acc1_initial : tensor<1x16x2x2xi32>) -> tensor<1x16x2x2xi32>
    %main_empty1 = tensor.empty() : tensor<1x16x2x2xui8>
    %main1 = standalone.conv_requant
      ins(%main_acc1, %multiplier, %shift4
          : tensor<1x16x2x2xi32>, tensor<16xi32>, tensor<16xi32>)
      outs(%main_empty1 : tensor<1x16x2x2xui8>)
      {output_zero_point = 100 : i32, relu_enable = 1 : i32}
      -> tensor<1x16x2x2xui8>

    %main_acc2_input_empty = tensor.empty() : tensor<1x16x2x2xi8>
    %main_acc2_input = linalg.generic {
      indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>, affine_map<(n, c, h, w) -> (n, c, h, w)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%main1 : tensor<1x16x2x2xui8>) outs(%main_acc2_input_empty : tensor<1x16x2x2xi8>) {
    ^bb0(%a: ui8, %unused: i8):
      %bits = builtin.unrealized_conversion_cast %a : ui8 to i8
      linalg.yield %bits : i8
    } -> tensor<1x16x2x2xi8>
    %main_acc2_zp = arith.constant 100 : i32
    %main_acc2_wzp = arith.constant 0 : i32
    %main_acc2_pad_value = arith.constant 100 : i8
    %main_acc2_padded = tensor.pad %main_acc2_input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%n: index, %c: index, %h: index, %w: index):
      tensor.yield %main_acc2_pad_value : i8
    } : tensor<1x16x2x2xi8> to tensor<1x16x4x4xi8>
    %main_acc2_empty = tensor.empty() : tensor<1x16x2x2xi32>
    %main_acc2_initial = linalg.broadcast ins(%bias : tensor<16xi32>) outs(%main_acc2_empty : tensor<1x16x2x2xi32>) dimensions = [0, 2, 3]
    %main_acc2 = linalg.conv_2d_nchw_fchw_q {
      strides = dense<[1, 1]> : tensor<2xi64>,
      dilations = dense<[1, 1]> : tensor<2xi64>,
      input_cast = #linalg.type_fn<cast_unsigned>,
      weight_cast = #linalg.type_fn<cast_signed>}
      ins(%main_acc2_padded, %main_weight2, %main_acc2_zp, %main_acc2_wzp : tensor<1x16x4x4xi8>, tensor<16x16x3x3xi8>, i32, i32)
      outs(%main_acc2_initial : tensor<1x16x2x2xi32>) -> tensor<1x16x2x2xi32>
    %main_empty2 = tensor.empty() : tensor<1x16x2x2xui8>
    %main2 = standalone.conv_requant
      ins(%main_acc2, %multiplier, %shift4
          : tensor<1x16x2x2xi32>, tensor<16xi32>, tensor<16xi32>)
      outs(%main_empty2 : tensor<1x16x2x2xui8>)
      {output_zero_point = 100 : i32, relu_enable = 0 : i32}
      -> tensor<1x16x2x2xui8>

    %skip_acc_input_empty = tensor.empty() : tensor<1x8x4x4xi8>
    %skip_acc_input = linalg.generic {
      indexing_maps = [affine_map<(n, c, h, w) -> (n, c, h, w)>, affine_map<(n, c, h, w) -> (n, c, h, w)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%input : tensor<1x8x4x4xui8>) outs(%skip_acc_input_empty : tensor<1x8x4x4xi8>) {
    ^bb0(%a: ui8, %unused: i8):
      %bits = builtin.unrealized_conversion_cast %a : ui8 to i8
      linalg.yield %bits : i8
    } -> tensor<1x8x4x4xi8>
    %skip_acc_zp = arith.constant 100 : i32
    %skip_acc_wzp = arith.constant 0 : i32
    %skip_acc_pad_value = arith.constant 100 : i8
    %skip_acc_padded = tensor.pad %skip_acc_input low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%n: index, %c: index, %h: index, %w: index):
      tensor.yield %skip_acc_pad_value : i8
    } : tensor<1x8x4x4xi8> to tensor<1x8x4x4xi8>
    %skip_acc_empty = tensor.empty() : tensor<1x16x2x2xi32>
    %skip_acc_initial = linalg.broadcast ins(%bias : tensor<16xi32>) outs(%skip_acc_empty : tensor<1x16x2x2xi32>) dimensions = [0, 2, 3]
    %skip_acc = linalg.conv_2d_nchw_fchw_q {
      strides = dense<[2, 2]> : tensor<2xi64>,
      dilations = dense<[1, 1]> : tensor<2xi64>,
      input_cast = #linalg.type_fn<cast_unsigned>,
      weight_cast = #linalg.type_fn<cast_signed>}
      ins(%skip_acc_padded, %skip_weight, %skip_acc_zp, %skip_acc_wzp : tensor<1x8x4x4xi8>, tensor<16x8x1x1xi8>, i32, i32)
      outs(%skip_acc_initial : tensor<1x16x2x2xi32>) -> tensor<1x16x2x2xi32>
    %skip_empty = tensor.empty() : tensor<1x16x2x2xui8>
    %skip = standalone.conv_requant
      ins(%skip_acc, %multiplier, %shift4
          : tensor<1x16x2x2xi32>, tensor<16xi32>, tensor<16xi32>)
      outs(%skip_empty : tensor<1x16x2x2xui8>)
      {output_zero_point = 100 : i32, relu_enable = 0 : i32}
      -> tensor<1x16x2x2xui8>

    %result_empty = tensor.empty() : tensor<1x16x2x2xui8>
    %result = standalone.qadd_relu
      ins(%main2, %skip
          : tensor<1x16x2x2xui8>, tensor<1x16x2x2xui8>)
      outs(%result_empty : tensor<1x16x2x2xui8>)
      {
        lhs_multiplier = 1 : i32,
        rhs_multiplier = 1 : i32,
        shift = 0 : i32,
        lhs_zero_point = 100 : i32,
        rhs_zero_point = 100 : i32,
        output_zero_point = 100 : i32,
        relu_enable = 1 : i32
      } -> tensor<1x16x2x2xui8>
    return %result : tensor<1x16x2x2xui8>
  }
}
