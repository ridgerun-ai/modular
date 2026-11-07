# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Covers the fused-epilogue closures inside the CPU conv wrappers.

`conv_nhwc_direct` and `conv_transposed_cpu` each build an internal
`ConvEpilogueFn` closure around the caller's SIMD epilogue. Those
two closures are reached only through op registration, so no other Mojo test
elaborates them; `test_conv_epilogue.mojo` calls `ConvDirectNHWC` directly and
bypasses both wrappers.
"""

from std.math import isclose, ceildiv
from std.random import rand
from std.sys import simd_width_of
from std.testing import assert_almost_equal

from layout import Coord, TileTensor, row_major
from nn.conv.conv import ConvInfoStatic, conv_nhwc_direct, pack_filter
from nn.conv.conv_transpose import conv_transposed_cpu
from nn.conv.conv_utils import (
    append_shape,
    extend_shape,
    get_direct_conv_micro_kernel_width,
)

from std.utils.index import Index, IndexList

comptime dtype = DType.float32
comptime SCALE = Scalar[dtype](3.0)


def test_conv_nhwc_direct_fused() raises:
    print("== test_conv_nhwc_direct_fused")

    comptime rank = 2
    var N = 1
    var C = 2
    var F = 4
    var input_dims = Index(5, 5)
    var filter_dims = Index(3, 3)
    var stride = Index(1, 1)
    var dilation = Index(1, 1)

    var output_dims = IndexList[rank](1)
    comptime for i in range(rank):
        output_dims[i] = input_dims[i] - filter_dims[i] + 1

    var input_size = N * input_dims[0] * input_dims[1] * C
    var input_ptr = List(length=input_size, fill=Scalar[dtype](0))
    rand(input_ptr)

    var filter_size = filter_dims[0] * filter_dims[1] * C * F
    var filter_ptr = List(length=filter_size, fill=Scalar[dtype](0))
    rand(filter_ptr)

    var output_size = N * output_dims[0] * output_dims[1] * F
    var plain_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var fused_ptr = List(length=output_size, fill=Scalar[dtype](0))

    var input_tt = TileTensor(
        input_ptr, row_major(extend_shape(Coord(input_dims), N, C))
    )
    var filter_tt = TileTensor(
        filter_ptr, row_major(extend_shape(Coord(filter_dims), C, F))
    )
    var plain_tt = TileTensor(
        plain_ptr, row_major(extend_shape(Coord(output_dims), N, F))
    )
    var fused_tt = TileTensor(
        fused_ptr, row_major(extend_shape(Coord(output_dims), N, F))
    )

    comptime conv_attr = ConvInfoStatic[rank]()

    var out_ptr = fused_tt.ptr
    var col_stride = F
    var row_stride = output_dims[1] * F
    var plane_stride = output_dims[0] * row_stride

    # Scale each computed element, so a coordinate the wrapper's internal
    # closure mis-translates lands on the wrong element and fails the compare.
    @inline(.always)
    def scale_epilogue[
        _dtype: DType, _width: SIMDLength, _alignment: Int = 1
    ](coords: Coord, val: SIMD[_dtype, _width]) {
        var out_ptr, var col_stride, var row_stride, var plane_stride
    }:
        comptime assert coords.rank == 4
        var flat = (
            Int(coords[0]) * plane_stride
            + Int(coords[1]) * row_stride
            + Int(coords[2]) * col_stride
            + Int(coords[3])
        )
        (out_ptr.unsafe_offset(flat)).unsafe_store(val.cast[dtype]() * SCALE)

    @inline(.always)
    def null_epilogue[
        _dtype: DType, _width: SIMDLength, _alignment: Int = 1
    ](coords: Coord, val: SIMD[_dtype, _width]):
        pass

    conv_nhwc_direct[
        dtype,
        dtype,
        dtype,
        False,
        conv_attr,
        False,
    ](
        input_tt,
        filter_tt,
        plain_tt,
        stride,
        dilation,
        Index(0, 0),
        Index(0, 0),
        Index(0, 0),
        1,
        null_epilogue,
    )

    conv_nhwc_direct[
        dtype,
        dtype,
        dtype,
        False,
        conv_attr,
        True,
    ](
        input_tt,
        filter_tt,
        fused_tt,
        stride,
        dilation,
        Index(0, 0),
        Index(0, 0),
        Index(0, 0),
        1,
        scale_epilogue,
    )

    for i in range(output_size):
        if not isclose(fused_ptr[i], plain_ptr[i] * SCALE):
            print("FAIL at", i, fused_ptr[i], plain_ptr[i] * SCALE)
            raise Error("conv_nhwc_direct fused epilogue mismatch")
    print("PASS")


def test_conv_transposed_cpu_fused() raises:
    print("== test_conv_transposed_cpu_fused")

    comptime rank = 2
    var N = 1
    var C = 2
    var F = 3
    var input_dims = Index(4, 4)
    var filter_dims = Index(3, 3)
    var stride = Index(1, 1)
    var dilation = Index(1, 1)

    var output_dims = IndexList[rank](1)
    comptime for i in range(rank):
        output_dims[i] = (
            (input_dims[i] - 1) * stride[i]
            + dilation[i] * (filter_dims[i] - 1)
            + 1
        )

    var input_size = N * input_dims[0] * input_dims[1] * C
    var input_ptr = List(length=input_size, fill=Scalar[dtype](0))
    rand(input_ptr)

    var filter_size = filter_dims[0] * filter_dims[1] * F * C
    var filter_ptr = List(length=filter_size, fill=Scalar[dtype](0))
    rand(filter_ptr)

    var output_size = N * output_dims[0] * output_dims[1] * F
    var plain_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var fused_ptr = List(length=output_size, fill=Scalar[dtype](0))

    var input_tt = TileTensor(
        input_ptr, row_major(extend_shape(Coord(input_dims), N, C))
    )
    var filter_tt = TileTensor(
        filter_ptr, row_major(append_shape(Coord(filter_dims), F, C))
    )
    var plain_tt = TileTensor(
        plain_ptr, row_major(extend_shape(Coord(output_dims), N, F))
    )
    var fused_tt = TileTensor(
        fused_ptr, row_major(extend_shape(Coord(output_dims), N, F))
    )

    var out_ptr = fused_tt.ptr
    var col_stride = F
    var row_stride = output_dims[1] * F
    var plane_stride = output_dims[0] * row_stride

    @inline(.always)
    def scale_epilogue[
        _dtype: DType, _width: SIMDLength, _alignment: Int = 1
    ](coords: Coord, val: SIMD[_dtype, _width]) {
        var out_ptr, var col_stride, var row_stride, var plane_stride
    }:
        comptime assert coords.rank == 4
        var flat = (
            Int(coords[0]) * plane_stride
            + Int(coords[1]) * row_stride
            + Int(coords[2]) * col_stride
            + Int(coords[3])
        )
        (out_ptr.unsafe_offset(flat)).unsafe_store(val.cast[dtype]() * SCALE)

    @inline(.always)
    def null_epilogue[
        _dtype: DType, _width: SIMDLength, _alignment: Int = 1
    ](coords: Coord, val: SIMD[_dtype, _width]):
        pass

    conv_transposed_cpu[False, False, False](
        plain_tt,
        input_tt,
        filter_tt,
        stride,
        dilation,
        Index(0, 0),
        Index(0, 0),
        Index(0, 0),
        null_epilogue,
    )

    conv_transposed_cpu[False, False, True](
        fused_tt,
        input_tt,
        filter_tt,
        stride,
        dilation,
        Index(0, 0),
        Index(0, 0),
        Index(0, 0),
        scale_epilogue,
    )

    for i in range(output_size):
        if not isclose(fused_ptr[i], plain_ptr[i] * SCALE):
            print("FAIL at", i, fused_ptr[i], plain_ptr[i] * SCALE)
            raise Error("conv_transposed_cpu fused epilogue mismatch")
    print("PASS")


def test_conv2d_reference[
    fused: Bool,
    packed: Bool,
](
    n: Int,
    h: Int,
    w: Int,
    c: Int,
    f: Int,
    r: Int,
    s: Int,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    groups: Int = 1,
) raises:
    var ho = (h + pad_h[0] + pad_h[1] - dilation[0] * (r - 1) - 1) // stride[
        0
    ] + 1
    var wo = (w + pad_w[0] + pad_w[1] - dilation[1] * (s - 1) - 1) // stride[
        1
    ] + 1
    var filter_c = c // groups
    var inputs = List(length=n * h * w * c, fill=Float32(0))
    var filters = List(length=r * s * filter_c * f, fill=Float32(0))
    var outputs = List(length=n * ho * wo * f, fill=Float32(0))
    for i in range(len(inputs)):
        inputs[i] = Float32(i % 17 - 8) / 17
    for i in range(len(filters)):
        filters[i] = Float32(i % 13 - 6) / 13
    var input = TileTensor(inputs, row_major((n, h, w, c)))
    var filter = TileTensor(filters, row_major((r, s, filter_c, f)))
    var output = TileTensor(outputs, row_major((n, ho, wo, f)))
    var out_ptr = output.ptr

    @inline(.always)
    def affine_epilogue[
        _dtype: DType, width: SIMDLength, alignment: Int = 1
    ](coords: Coord, value: SIMD[_dtype, width]) {imm}:
        var batch = Int(coords[0])
        var y = Int(coords[1])
        var x = Int(coords[2])
        var channel = Int(coords[3])
        var offset = ((batch * ho + y) * wo + x) * f + channel
        # Coordinates matter: exercise translation back from matrix rows.
        var bias = Float32(batch + y + x)
        (out_ptr.unsafe_offset(offset)).unsafe_store(
            (value * 2 + SIMD[_dtype, width](bias)).cast[dtype]()
        )

    comptime if packed:
        comptime block = get_direct_conv_micro_kernel_width() * simd_width_of[
            dtype
        ]()
        var packed_filters = List(
            length=groups
            * ceildiv(f // groups, block)
            * r
            * s
            * filter_c
            * block,
            fill=Float32(0),
        )
        var packed_filter = TileTensor(
            packed_filters,
            row_major(
                (groups * ceildiv(f // groups, block), r, s, filter_c, block)
            ),
        )
        pack_filter(filter, packed_filter, groups)
        conv_nhwc_direct[
            dtype,
            dtype,
            dtype,
            True,
            ConvInfoStatic[2](),
            fused,
        ](
            input,
            packed_filter,
            output,
            stride,
            dilation,
            Index(0, 0),
            pad_h,
            pad_w,
            groups,
            affine_epilogue,
        )
    else:
        conv_nhwc_direct[
            dtype,
            dtype,
            dtype,
            False,
            ConvInfoStatic[2](),
            fused,
        ](
            input,
            filter,
            output,
            stride,
            dilation,
            Index(0, 0),
            pad_h,
            pad_w,
            groups,
            affine_epilogue,
        )
    for batch in range(n):
        for y in range(ho):
            for x in range(wo):
                for channel in range(f):
                    var expected = Float32(0)
                    for ky in range(r):
                        var iy = y * stride[0] + ky * dilation[0] - pad_h[0]
                        for kx in range(s):
                            var ix = x * stride[1] + kx * dilation[1] - pad_w[0]
                            if iy >= 0 and iy < h and ix >= 0 and ix < w:
                                for ic in range(filter_c):
                                    expected += (
                                        input[
                                            batch,
                                            iy,
                                            ix,
                                            channel // (f // groups) * filter_c
                                            + ic,
                                        ]
                                        * filter[ky, kx, ic, channel]
                                    )
                    comptime if fused:
                        expected = expected * 2 + Float32(batch + y + x)
                    assert_almost_equal(
                        output[batch, y, x, channel],
                        expected,
                        atol=1e-4,
                        rtol=1e-4,
                    )


def test_conv2d_reference_cases() raises:
    print("== test_conv2d_reference_cases")
    comptime for mode in range(4):
        comptime fused = Bool(mode % 2)
        comptime packed = Bool(mode // 2)
        # Tiny, padded, non-SIMD channel tails.
        test_conv2d_reference[fused, packed](
            1,
            1,
            1,
            33,
            3,
            3,
            3,
            Index(1, 1),
            Index(1, 1),
            Index(1, 1),
            Index(1, 1),
        )
        # Multiple batches, asymmetric padding, rectangular filter, stride.
        test_conv2d_reference[fused, packed](
            2,
            9,
            11,
            35,
            7,
            2,
            3,
            Index(2, 1),
            Index(1, 1),
            Index(1, 0),
            Index(2, 1),
        )
        # Dilation with independent spatial strides.
        test_conv2d_reference[fused, packed](
            2,
            13,
            15,
            40,
            17,
            3,
            3,
            Index(1, 2),
            Index(2, 2),
            Index(2, 2),
            Index(2, 2),
        )
        # More than one scratch tile, including the final partial tile.
        test_conv2d_reference[fused, packed](
            2,
            33,
            35,
            32,
            65,
            3,
            3,
            Index(1, 1),
            Index(1, 1),
            Index(1, 1),
            Index(1, 1),
        )
        comptime if packed:
            # Grouped filters keep the existing direct implementation.
            test_conv2d_reference[fused, packed](
                2,
                7,
                9,
                64,
                12,
                3,
                3,
                Index(1, 1),
                Index(1, 1),
                Index(1, 1),
                Index(1, 1),
                2,
            )
        # 1x1 remains on the existing direct path.
        test_conv2d_reference[fused, packed](
            1,
            5,
            7,
            3,
            5,
            1,
            1,
            Index(1, 1),
            Index(1, 1),
            Index(0, 0),
            Index(0, 0),
        )

    print("PASS (22 reference cases)")


def main() raises:
    test_conv_nhwc_direct_fused()
    test_conv2d_reference_cases()
    test_conv_transposed_cpu_fused()
